import torch
import torch.nn as nn
from tbsim.models.base_models import *
from collections import defaultdict
import torch.nn.functional as F




class clique_guided_policy_net(nn.Module):
    def __init__(self,node_types,edge_types, z_dim, rel_state_fun, collision_fun, dyn_net, edge_encoding_net, edge_enc_dim , map_enc_dim,
                  history_enc_dim, obs_lstm_hidden_dim,guide_RNN_hidden_dim, FC_hidden_dim, max_Nnode, dt, att_internal_dim = 16):
                 
        super(clique_guided_policy_net, self).__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = {nt:dyn_net[nt].udim for nt in dyn_net}
        self.state_dim = {nt:dyn_net[nt].xdim for nt in dyn_net}
        self.z_dim = z_dim
        self.dyn_net = dyn_net
        self.max_Nnode = max_Nnode
        self.guide_RNN_hidden_dim = guide_RNN_hidden_dim
        self.obs_lstm_hidden_dim = obs_lstm_hidden_dim
        self.history_enc_dim = history_enc_dim
        self.edge_encoding_net = edge_encoding_net
        self.edge_enc_dim = edge_enc_dim
        self.rel_state_fun = rel_state_fun
        self.collision_fun = collision_fun
        self.dt = dt
        if map_enc_dim is None:
            self.map_encoding = False
        else:
            self.map_encoding = True
            self.map_enc_dim  =map_enc_dim

        self.obs_lstm_h0_net = nn.ModuleDict()
        self.obs_lstm_c0_net = nn.ModuleDict()
        self.guide_hidden_net = nn.ModuleDict()
        self.guide_RNN = nn.ModuleDict()
        self.RNN_proj_net = nn.ModuleDict()

        self.obs_att = nn.ModuleDict()
        self.obs_lstm = nn.ModuleDict()
        self.state_lstm = nn.ModuleDict()
        self.action_net = nn.ModuleDict()
        for node_type in self.node_types:
            self.obs_att[node_type.name] = AdditiveAttention(encoder_hidden_state_dim=edge_enc_dim, decoder_hidden_state_dim=self.state_dim[node_type], internal_dim=att_internal_dim)
            self.obs_lstm[node_type.name] = nn.LSTM(edge_enc_dim,obs_lstm_hidden_dim)
            
            self.obs_lstm_h0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.obs_lstm_c0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.action_net[node_type.name] = MLP(obs_lstm_hidden_dim+self.z_dim+self.state_dim[node_type]+4,self.input_dim[node_type],FC_hidden_dim)

            self.RNN_proj_net[node_type.name] = nn.Linear(self.guide_RNN_hidden_dim,2)
            if self.map_encoding  and node_type in self.map_enc_dim:
                if self.algo_config['use_lane_dec']:
                    self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type]+3,self.guide_RNN_hidden_dim)
                    self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
                else:
                    self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
                    self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
            else:
                self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim,self.guide_RNN_hidden_dim)
                self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim,self.guide_RNN_hidden_dim)
            
    def forward(self, batch_state_history,batch_state_history_st,node_history_encoded,encoded_map,batch_node_size,batch_lane_st, indices, ft, batch_z,robot_traj = None):
        node_index,edge_index,node_inverse_index,batch_node_to_edge_index,edge_to_node_index,batch_edge_idx1,batch_edge_idx2 = indices

        batch_state_pred = dict()
        batch_input_pred = dict()
        batch_state_pred_st = dict()
        batch_edge = dict()
        batch_obs = dict()
        obs_lstm_h = dict()
        obs_lstm_c = dict()
        batch_edge_enc = [None]*ft

        node_num = 0 
        for nt in self.node_types:
            node_num+=len(node_index[nt])
        tracking_error = 0
        collision_cost = 0
        for node_type in self.node_types:
            batch_state_pred[node_type] = [None]*ft
            batch_state_pred_st[node_type] = [None]*ft
            batch_input_pred[node_type] = [None]*ft

        
        edge_to_obs_idx = dict()
        for et in self.edge_types:
            edge_to_obs_idx[et] = defaultdict(list)
            for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
                edge_to_obs_idx[et][node_idx].append(idx)

        device = batch_state_history_st.device

        # for edge_type in self.edge_types:
        #     batch_edge[edge_type] = torch.zeros([len(edge_index[edge_type]),self.state_dim[edge_type[0]]+self.state_dim[edge_type[1]]])

        des_traj = dict()
        for nt in self.node_types:

            obs_lstm_h[nt] = self.obs_lstm_h0_net[nt](batch_state_history_st[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(device)
            obs_lstm_c[nt] = self.obs_lstm_c0_net[nt](batch_state_history_st[nt][-1]).view(1,len(node_index[nt]),self.obs_lstm_hidden_dim).to(device)
            

            if self.map_encoding and nt in self.map_enc_dim:
                xz = torch.cat((node_history_encoded[nt][-1],encoded_map[nt],batch_z[nt]),dim=-1)
            else:
                xz = torch.cat((node_history_encoded[nt][-1],batch_z[nt]),dim=-1)
            
            if self.algo_config['use_lane_dec'] and self.map_encoding and nt in self.map_enc_dim:
                # guide_RNN_hidden = self.guide_hidden_net[nt](torch.unsqueeze(xz,0))
                # des_traj[nt] = list()
                # for t in range(ft):

                #     if t==0:
                #         wp = torch.zeros([xz.shape[0],3]).to(self.device)
                #     elif t==1:
                        
                #         psi = torch.atan2(des_traj[nt][0][:,1],des_traj[nt][0][:,0]).reshape(-1,1)
                #         wp = torch.cat((des_traj[nt][0],psi),dim=-1)
                #     else:
                #         psi = torch.atan2(des_traj[nt][t-1][:,1]-des_traj[nt][t-2][:,1],des_traj[nt][t-1][:,0]-des_traj[nt][t-2][:,0]).reshape(-1,1)
                #         wp = torch.cat((des_traj[nt][t-1],psi),dim=-1)
                #     delta_y,delta_psi,ref_pt = batch_proj(wp,batch_lane_st[nt][...,[0,1,3]].permute(1,0,2))
                #     if delta_y.isnan().any() or delta_psi.isnan().any() or ref_pt.isnan().any():
                #         pdb.set_trace()
                #     ref_psi = ref_pt[...,2:3]
                #     RNN_input = torch.unsqueeze(torch.cat((xz,delta_y,torch.cos(ref_psi),torch.sin(ref_psi)),dim=-1),dim=0)
                #     RNN_out,guide_RNN_hidden = self.guide_RNN[nt](RNN_input,guide_RNN_hidden)
                #     des_traj[nt].append((self.RNN_proj_net[nt](RNN_out[0])))
                
                # des_traj[nt] = torch.stack(des_traj[nt],dim=0)
                xz = torch.unsqueeze(xz,0)
                xz_seq = torch.cat((xz.repeat(ft,1,1),batch_lane_st[nt][...,[0,1,3]]),dim=-1)
                guide_RNN_hidden = self.guide_hidden_net[nt](xz)
                RNN_out,_ = self.guide_RNN[nt](xz_seq,guide_RNN_hidden)
                des_traj[nt] = self.RNN_proj_net[nt](RNN_out)

            else:
                xz = torch.unsqueeze(xz,0)
                xz_seq = xz.repeat(ft,1,1)
                guide_RNN_hidden = self.guide_hidden_net[nt](xz)
                RNN_out,_ = self.guide_RNN[nt](xz_seq,guide_RNN_hidden)
                des_traj[nt] = self.RNN_proj_net[nt](RNN_out)

        
        node_obs_idx = {nt:torch.zeros([len(node_index[nt]),self.max_Nnode-1],dtype=torch.long) for nt in self.node_types}
        offset = 1
        edge_idx_offset = dict()

        for et in self.edge_types:
            edge_idx_offset[et] = offset
            offset+= len(edge_index[et])
        for et in self.edge_types:
            nt = et[0]
            for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
                node_obs_idx[nt][node_idx,nb_idx] = idx + edge_idx_offset[et]


        for t in range(ft):
            batch_obs = dict()

            ## put states into raw obs tensor
            batch_edge_enc[t] = torch.zeros([1,self.edge_enc_dim]).to(device)
            for edge_type in self.edge_types:
                dim1 = self.state_dim[edge_type[0]]
                dim2 = self.state_dim[edge_type[1]]
                try:
                    edge_node_size1 = batch_node_size[edge_type[0]][batch_edge_idx1[edge_type]]
                    edge_node_size2 = batch_node_size[edge_type[1]][batch_edge_idx2[edge_type]]
                except:

                    edge_node_size1 = batch_node_size[edge_type[0]][batch_edge_idx1[edge_type]]
                    edge_node_size2 = batch_node_size[edge_type[1]][batch_edge_idx2[edge_type]]

                if t==0:
                    batch_edge[edge_type] = torch.cat((batch_state_history[edge_type[0]][-1,batch_edge_idx1[edge_type]],\
                                                       batch_state_history[edge_type[1]][-1,batch_edge_idx2[edge_type]]),dim=1) 
                else:

                    batch_edge[edge_type] = torch.cat((batch_state_pred[edge_type[0]][t-1][batch_edge_idx1[edge_type]],\
                                                       batch_state_pred[edge_type[1]][t-1][batch_edge_idx2[edge_type]]),dim=1)

                if batch_edge[edge_type].shape[0]>0:
                    collision_cost += torch.sum(self.collision_fun[edge_type](batch_edge[edge_type][:,0:dim1],batch_edge[edge_type][:,dim1:dim1+dim2],edge_node_size1,edge_node_size2))
            ## pass through pre-encoding network
                if batch_edge[edge_type].shape[0]>0:
                    batch_edge_enc[t] = torch.cat((batch_edge_enc[t],self.edge_encoding_net[edge_type](batch_edge[edge_type][:,0:dim1],batch_edge[edge_type][:,dim1:dim1+dim2],edge_node_size1,edge_node_size2)),dim=0)
                    if batch_edge_enc[t].isnan().any():
                        import pdb
                        pdb.set_trace()
            ## put encoded vectors into observation matrices of each node
            for nt in self.node_types:
                if node_obs_idx[nt].shape[0]>0:
                    # if (node_obs_idx[nt]==0).all():
                    #     pdb.set_trace()
                    try:
                        batch_obs[nt] = batch_edge_enc[t][node_obs_idx[nt]]
                    except:
                        batch_obs[nt] = batch_edge_enc[t][node_obs_idx[nt]]


            ## pass the observations through attention network and LSTM, and eventually action net
            for nt in self.node_types:
                if len(node_index[nt])>0:
                    if t==0:
                        rel_state = batch_state_history_st[nt][-1]
                        next_wp = des_traj[nt][t]-batch_state_history_st[nt][-1][:,0:2]
                    else:
                        rel_state = self.rel_state_fun[nt](batch_state_pred_st[nt][t-1],batch_state_history_st[nt][-1])
                        rel_state[...,0:2]-=des_traj[nt][t-1]
                        next_wp = des_traj[nt][t]-batch_state_pred_st[nt][t-1][:,0:2]

                    tracking_error += torch.linalg.norm(rel_state[...,0:2])/rel_state.shape[0]

                    batch_obs_enc,_ = self.obs_att[nt](batch_obs[nt],rel_state)
                    if batch_obs_enc.isnan().any():
                        pdb.set_trace()
                    obs_lstm_out,(obs_lstm_h[nt],obs_lstm_c[nt]) = self.obs_lstm[nt](torch.unsqueeze(batch_obs_enc,dim=0),(obs_lstm_h[nt],obs_lstm_c[nt]))
                    
                    if obs_lstm_out.isnan().any():
                        pdb.set_trace()
                    batch_input_pred[nt][t] = self.action_net[nt](torch.cat((rel_state,next_wp,batch_node_size[nt],torch.squeeze(obs_lstm_out,dim=0),batch_z[nt]),dim=-1))

                    if not robot_traj is None:
                        for idx,_ in robot_traj[nt].items():
                            batch_input_pred[nt][t][idx]=torch.zeros(self.input_dim[nt]).to(device)

                    ## integrate forward the dynamics
                    if t==0:
                        batch_state_pred[nt][t] = self.dyn_net[nt](batch_state_history[nt][-1],batch_input_pred[nt][t],self.dt)
                        batch_state_pred_st[nt][t] = self.dyn_net[nt](batch_state_history_st[nt][-1],batch_input_pred[nt][t],self.dt)
                    else:
                        batch_state_pred[nt][t] = self.dyn_net[nt](batch_state_pred[nt][t-1],batch_input_pred[nt][t],self.dt)
                        batch_state_pred_st[nt][t] = self.dyn_net[nt](batch_state_pred_st[nt][t-1],batch_input_pred[nt][t],self.dt)

                    if batch_state_pred[nt][t].isnan().any() or batch_state_pred_st[nt][t].isnan().any():
                        pdb.set_trace()

                    if not robot_traj is None:
                        for idx,(traj,traj_st) in robot_traj[nt].items():
                            if not (traj[t]==0).all():
                                batch_state_pred[nt][t][idx] = traj[t]
                                batch_state_pred_st[nt][t][idx] = traj_st[t]
                else:
                    batch_state_pred[nt][t] = torch.zeros([0,self.state_dim[nt]]).to(device)
                    batch_state_pred_st[nt][t] = torch.zeros([0,self.state_dim[nt]]).to(device)
                    batch_input_pred[nt][t] = torch.zeros([0,self.input_dim[nt]]).to(device)
            

        for nt in self.node_types:

            batch_state_pred[nt] = torch.stack(batch_state_pred[nt])
            batch_state_pred_st[nt] = torch.stack(batch_state_pred_st[nt])
            batch_input_pred[nt] = torch.stack(batch_input_pred[nt])

        return batch_state_pred,batch_state_pred_st,batch_input_pred,des_traj,tracking_error,collision_cost/node_num

 
class guided_policy_net(nn.Module):
    def __init__(self,node_types,edge_types, z_dim, rel_state_fun, collision_fun, dyn_net, edge_encoding_net, edge_enc_dim , map_enc_dim,
                  history_enc_dim, obs_lstm_hidden_dim,guide_RNN_hidden_dim, FC_hidden_dim, max_Nnode, dt, use_lane_dec=False, att_internal_dim = 16,output_var = False):
                 
        super(guided_policy_net, self).__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = {nt:dyn_net[nt].udim for nt in dyn_net}
        self.state_dim = {nt:dyn_net[nt].xdim for nt in dyn_net}
        self.z_dim = z_dim
        self.dyn_net = dyn_net
        self.max_Nnode = max_Nnode
        self.guide_RNN_hidden_dim = guide_RNN_hidden_dim
        self.obs_lstm_hidden_dim = obs_lstm_hidden_dim
        self.history_enc_dim = history_enc_dim
        self.edge_encoding_net = edge_encoding_net
        self.edge_enc_dim = edge_enc_dim
        self.rel_state_fun = rel_state_fun
        self.collision_fun = collision_fun
        self.use_lane_dec = use_lane_dec
        self.dt = dt
        self.output_var = output_var
        if map_enc_dim is None:
            self.map_encoding = False
        else:
            self.map_encoding = True
            self.map_enc_dim  =map_enc_dim

        self.obs_lstm_h0_net = nn.ModuleDict()
        self.obs_lstm_c0_net = nn.ModuleDict()
        self.guide_hidden_net = nn.ModuleDict()
        self.guide_RNN = nn.ModuleDict()
        self.RNN_proj_net = nn.ModuleDict()

        self.obs_att = nn.ModuleDict()
        self.obs_lstm = nn.ModuleDict()
        self.state_lstm = nn.ModuleDict()
        self.action_net = nn.ModuleDict()
        for node_type in self.node_types:
            self.obs_att[node_type.name] = AdditiveAttention(encoder_hidden_state_dim=edge_enc_dim, decoder_hidden_state_dim=self.state_dim[node_type], internal_dim=att_internal_dim)
            self.obs_lstm[node_type.name] = nn.LSTM(edge_enc_dim,obs_lstm_hidden_dim)
            
            self.obs_lstm_h0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            self.obs_lstm_c0_net[node_type.name] = nn.Linear(self.state_dim[node_type],self.obs_lstm_hidden_dim)
            output_dim = self.input_dim[node_type]+2 if self.output_var else self.input_dim[node_type]
            self.action_net[node_type.name] = MLP(obs_lstm_hidden_dim+self.z_dim+self.state_dim[node_type]+4,output_dim,FC_hidden_dim)

            self.RNN_proj_net[node_type.name] = nn.Linear(self.guide_RNN_hidden_dim,2)
            if self.map_encoding  and node_type in self.map_enc_dim:
                if self.use_lane_dec:
                    self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type]+3,self.guide_RNN_hidden_dim)
                    self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
                else:
                    self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
                    self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim+self.map_enc_dim[node_type],self.guide_RNN_hidden_dim)
            else:
                self.guide_RNN[node_type.name] = nn.GRU(self.history_enc_dim+self.z_dim,self.guide_RNN_hidden_dim)
                self.guide_hidden_net[node_type.name] = nn.Linear(self.history_enc_dim+self.z_dim,self.guide_RNN_hidden_dim)
    
    def get_rel_state(self,x1,x2,node_types):
        rel_state = [self.rel_state_fun[nt](x1,x2)*(node_types==nt.value).unsqueeze(-1) for nt in self.node_types]
        return sum(rel_state)


    def forward(self, 
                state_history,
                state_history_local,   
                node_history_encoded,
                encoded_map,
                node_size, 
                lane_st, 
                Tf, 
                z_clique,
                z_node,
                clique_index, 
                node_types,
                cond=None):

        bs,Na = state_history.shape[:2]
        Nc,numMode = z_clique.shape[1:3]
        tracking_error = 0
        collision_cost = 0

        state_pred = [None]*Tf
        state_pred_local = [None]*Tf
        input_pred = [None]*Tf
        logvar = [None]*Tf
        
        x0 = state_history[...,-1,:]
        x0_tiled = x0.unsqueeze(-2).repeat_interleave(numMode,-2)
        x0_local = state_history_local[...,-1,:]
        x0_local_tiled = x0_local.unsqueeze(-2).repeat_interleave(numMode,-2)
        node_size_tiled = node_size.unsqueeze(2).repeat_interleave(numMode,2)
        z_node_onehot = F.one_hot(z_node,self.z_dim)
        Th = state_history.shape[-2]


        device = state_history.device

        # for edge_type in self.edge_types:
        #     batch_edge[edge_type] = torch.zeros([len(edge_index[edge_type]),self.state_dim[edge_type[0]]+self.state_dim[edge_type[1]]])

        des_traj = dict()
        obs_lstm_h = list()
        obs_lstm_c = list()
        obs_lstm_h = {nt:TensorUtils.join_dimensions(self.obs_lstm_h0_net[nt](x0_local_tiled),0,3).unsqueeze(0) for nt in self.node_types}
        obs_lstm_c = {nt:TensorUtils.join_dimensions(self.obs_lstm_c0_net[nt](x0_local_tiled),0,3).unsqueeze(0) for nt in self.node_types}

        
        xz = torch.cat((node_history_encoded[:,:,-1:].repeat_interleave(numMode,-2),encoded_map.unsqueeze(-2).repeat_interleave(numMode,-2),z_node_onehot),dim=-1)
        
        # if self.algo_config['use_lane_dec'] and self.map_encoding and nt in self.map_enc_dim:
        #     # guide_RNN_hidden = self.guide_hidden_net[nt](torch.unsqueeze(xz,0))
        #     # des_traj[nt] = list()
        #     # for t in range(ft):

        #     #     if t==0:
        #     #         wp = torch.zeros([xz.shape[0],3]).to(self.device)
        #     #     elif t==1:
                    
        #     #         psi = torch.atan2(des_traj[nt][0][:,1],des_traj[nt][0][:,0]).reshape(-1,1)
        #     #         wp = torch.cat((des_traj[nt][0],psi),dim=-1)
        #     #     else:
        #     #         psi = torch.atan2(des_traj[nt][t-1][:,1]-des_traj[nt][t-2][:,1],des_traj[nt][t-1][:,0]-des_traj[nt][t-2][:,0]).reshape(-1,1)
        #     #         wp = torch.cat((des_traj[nt][t-1],psi),dim=-1)
        #     #     delta_y,delta_psi,ref_pt = batch_proj(wp,batch_lane_st[nt][...,[0,1,3]].permute(1,0,2))
        #     #     if delta_y.isnan().any() or delta_psi.isnan().any() or ref_pt.isnan().any():
        #     #         pdb.set_trace()
        #     #     ref_psi = ref_pt[...,2:3]
        #     #     RNN_input = torch.unsqueeze(torch.cat((xz,delta_y,torch.cos(ref_psi),torch.sin(ref_psi)),dim=-1),dim=0)
        #     #     RNN_out,guide_RNN_hidden = self.guide_RNN[nt](RNN_input,guide_RNN_hidden)
        #     #     des_traj[nt].append((self.RNN_proj_net[nt](RNN_out[0])))
            
        #     # des_traj[nt] = torch.stack(des_traj[nt],dim=0)
        #     xz = torch.unsqueeze(xz,0)
        #     xz_seq = torch.cat((xz.repeat_interleave(Tf,0),lane_st[...,[0,1,3]]),dim=-1)
        #     guide_RNN_hidden = self.guide_hidden_net[nt](xz)
        #     RNN_out,_ = self.guide_RNN[nt](xz_seq,guide_RNN_hidden)
        #     des_traj[nt] = self.RNN_proj_net[nt](RNN_out)

        # else:
        #     xz = torch.unsqueeze(xz,0)
        #     xz_seq = xz.repeat(ft,1,1)
        #     guide_RNN_hidden = self.guide_hidden_net[nt](xz)
        #     RNN_out,_ = self.guide_RNN[nt](xz_seq,guide_RNN_hidden)
        #     des_traj[nt] = self.RNN_proj_net[nt](RNN_out)

        xz = torch.unsqueeze(xz,0)
        xz_seq = xz.repeat_interleave(Tf,0)
        guide_RNN_hidden = {nt:self.guide_hidden_net[nt](xz) for nt in self.node_types}
        des_traj = list()
        for nt in self.node_types:
            RNN_out,_ = self.guide_RNN[nt](TensorUtils.join_dimensions(xz_seq,1,4),TensorUtils.join_dimensions(guide_RNN_hidden[nt],1,4))
            des_traj.append(self.RNN_proj_net[nt](RNN_out).reshape(Tf,bs,Na,-1,2)*(node_types==nt.value)[None,:,:,None,None])
        des_traj = sum(des_traj)

        
        # node_obs_idx = {nt:torch.zeros([len(node_index[nt]),self.max_Nnode-1],dtype=torch.long) for nt in self.node_types}
        # offset = 1
        # edge_idx_offset = dict()

        # for et in self.edge_types:
        #     edge_idx_offset[et] = offset
        #     offset+= len(edge_index[et])
        # for et in self.edge_types:
        #     nt = et[0]
        #     for idx, (node_idx,nb_idx) in edge_to_node_index[et].items():
        #         node_obs_idx[nt][node_idx,nb_idx] = idx + edge_idx_offset[et]

        edge_enc = [None]*Tf
        if cond is not None:
            cond_state,cond_state_local = cond
        for t in range(Tf):
            # if t==0:
            #     batch_edge = torch.cat((state_history[...,-1,:].unsqueeze(2).repeat_interleave(Na,2),\
            #                                         state_history[...,-1,:].unsqueeze(1).repeat_interleave(Na,1)),dim=-1) 
            # else:
            #     batch_edge = torch.cat((state_pred[t-1].unsqueeze(2).repeat_interleave(Na,2),\
            #                                         state_pred[t-1].unsqueeze(1).repeat_interleave(Na,1)),dim=1)
            ## put states into raw obs tensor
            edge_enc[t] = list()
            self_mask = ~torch.eye(Na,dtype=torch.bool).unsqueeze(0).to(device)
            for et in self.edge_types:
        
            ## pass through pre-encoding network
                et_mask = (node_types==et[0].value).unsqueeze(2).repeat_interleave(Na,2)*(node_types==et[0].value).unsqueeze(1).repeat_interleave(Na,1)
                if t==0:
                    
                    edge_et = self.edge_encoding_net[et](state_history[...,-1,:].unsqueeze(2).repeat_interleave(Na,2),
                                                         state_history[...,-1,:].unsqueeze(1).repeat_interleave(Na,1),
                                                         node_size.unsqueeze(2).repeat_interleave(Na,2),
                                                         node_size.unsqueeze(1).repeat_interleave(Na,1))
                    edge_et = edge_et.unsqueeze(3).repeat_interleave(numMode,3)
                    edge_enc[t].append(edge_et*(et_mask*self_mask)[...,None,None])
                else:
                    edge_enc[t].append(self.edge_encoding_net[et](state_pred[t-1].unsqueeze(2).repeat_interleave(Na,2),
                                                                  state_pred[t-1].unsqueeze(1).repeat_interleave(Na,1),
                                                                  node_size[:,:,None,None,:].repeat_interleave(Na,2).repeat_interleave(numMode,3),
                                                                  node_size[:,None,:,None,:].repeat_interleave(Na,1).repeat_interleave(numMode,3))*(et_mask*self_mask)[...,None,None])

            edge_enc[t] = sum(edge_enc[t])  
            clique_mask = clique_index.unsqueeze(2).repeat_interleave(Na,2)==clique_index.unsqueeze(1).repeat_interleave(Na,1)
            clique_mask *= ~torch.eye(Na,dtype=torch.bool,device=device).unsqueeze(0).repeat_interleave(bs,0)
            edge_enc[t] *= clique_mask[...,None,None]





            ## pass the observations through attention network and LSTM, and eventually action net
            if t==0:
                rel_state = x0_local_tiled
                next_wp = des_traj[t]-x0_local_tiled[...,:2]
            else:
                rel_state = sum([self.rel_state_fun[nt](state_pred_local[t-1],x0_local_tiled)*(node_types==nt.value)[...,None,None] for nt in self.node_types])
                rel_state[...,0:2]-=des_traj[t-1]
                next_wp = des_traj[t]-state_pred_local[t-1][...,0:2]
            tracking_error += torch.linalg.norm(rel_state[...,0:2])/math.prod(rel_state.shape[:-1])
            input_pred[t] = list()
            logvar[t] = list()
            for nt in self.node_types:
                
                obs_enc,_ = self.obs_att[nt](TensorUtils.join_dimensions(edge_enc[t].transpose(2,3),0,3),TensorUtils.join_dimensions(rel_state,0,3))
                obs_lstm_out,(obs_lstm_h[nt],obs_lstm_c[nt]) = self.obs_lstm[nt](obs_enc.unsqueeze(0),(obs_lstm_h[nt],obs_lstm_c[nt]))
                output = self.action_net[nt](torch.cat((rel_state,next_wp,node_size_tiled,obs_lstm_out.view(bs,Na,numMode,-1),z_node_onehot),dim=-1))
                if self.output_var:
                    input_pred_nt,logvar_nt = torch.split(output,[self.input_dim[nt],2],dim=-1)
                else:
                    input_pred_nt = output
                input_pred[t].append(input_pred_nt*(node_types==nt.value)[...,None,None])
                logvar[t].append(logvar_nt*(node_types==nt.value)[...,None,None])
            input_pred[t] = sum(input_pred[t])
            logvar[t] = sum(logvar[t])

            ## integrate forward the dynamics
            if t==0:
                state_pred[t] = sum([self.dyn_net[nt].step(x0_tiled,input_pred[t],self.dt)*(node_types==nt.value)[...,None,None] for nt in self.node_types])
                state_pred_local[t] = sum([self.dyn_net[nt].step(x0_local_tiled,input_pred[t],self.dt)*(node_types==nt.value)[...,None,None] for nt in self.node_types])
            else:
                state_pred[t] = sum([self.dyn_net[nt].step(state_pred[t-1],input_pred[t],self.dt)*(node_types==nt.value)[...,None,None] for nt in self.node_types])
                state_pred_local[t] = sum([self.dyn_net[nt].step(state_pred_local[t-1],input_pred[t],self.dt)*(node_types==nt.value)[...,None,None] for nt in self.node_types])

            if cond is not None:
                cond_flag = (cond_state!=0).any(-1).any(-1)
                state_pred[t] = state_pred[t]*(~cond_flag[...,None,None]) + cond_state[...,t:t+1,:]*cond_flag[...,None,None]
                state_pred_local[t] = state_pred_local[t]*(~cond_flag[...,None,None]) + cond_state_local[...,t:t+1,:]*cond_flag[...,None,None]

        state_pred = torch.stack(state_pred,3)
        state_pred_local = torch.stack(state_pred_local,3)
        input_pred = torch.stack(input_pred,3)
        if self.output_var:
            logvar = torch.stack(logvar,3)
            var = torch.exp(logvar)
        else:
            var = None
        return state_pred,state_pred_local,input_pred,des_traj.permute(1,2,3,0,4),var,tracking_error