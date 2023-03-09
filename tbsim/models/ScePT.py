from os import DirEntry
from tkinter import N
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum

from tbsim.models.base_models import *
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation, DynOrnsteinUhlenbeckPerturbation
from tbsim.utils.geometry_utils import *
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.model_utils as ModelUtils
from tbsim.models.policy_net import guided_policy_net
from tbsim.utils.loss_utils import (
    trajectory_loss,
    MultiModal_trajectory_loss,
    goal_reaching_loss,
    collision_loss,
    collision_loss_masked,
    log_normal_mixture,
    NLL_GMM_loss,
    compute_pred_loss,
    diversity_score,
    KLD_discrete_with_zero,
    KLD_discrete
)
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

import itertools
import sys
# from utils import CVaR, CVaR_weight
from functools import partial

thismodule = sys.modules[__name__]

class NodeType(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == str and self.name == other:
            return True
        else:
            return isinstance(other, self.__class__) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        return self.name + other

PED = NodeType("PEDESTRIAN",1)
VEH = NodeType("VEHICLE",3)



class ScenePolicyTrajectron(nn.Module):
    def __init__(
        self,
        algo_config,
        modality_shapes,
        node_types=[VEH],
        edge_types=[(VEH,VEH)],
        weights_scaling = None,
    ):
        """
        node_types: list of node_types ('VEHICLE', 'PEDESTRIAN'...)
        edge_types: list of edge types
        """
        super(ScenePolicyTrajectron, self).__init__()
        self.algo_config = algo_config
        self.node_types = node_types
        self.edge_types = edge_types
        self.curr_iter = 0
        self.z_dim = algo_config.latent_dim
        self.edge_pre_enc_net = dict()
        self.node_modules = nn.ModuleDict()
        self.modality_shapes = modality_shapes

        self.min_hl = 1
        self.max_hl = algo_config.history_num_frames
        self.ph = algo_config.future_num_frames

        self.dynamic = dict()
        self.rel_state_fun = dict()
        self.collision_fun = dict()
        if weights_scaling is None:
            weights_scaling=[1,1,1]
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        self.dt = algo_config.step_time
        self.ego_conditioning = algo_config.ego_conditioning
        self.gamma = algo_config.gamma
        self.stage = algo_config.stage
        self.num_frames_per_stage = algo_config.num_frames_per_stage
        self.output_var = algo_config.output_var
        self.UAC = algo_config.UAC
        
        
        # relative state function, generate translation invariant relative state between nodes
        self.rel_state_fun[PED] = ModelUtils.PED_rel_state
        self.rel_state_fun[VEH] = ModelUtils.VEH_rel_state

        # collision function between different types of nodes, return distance or collision penalty.
        self.collision_fun[(PED,PED)] = PED_PED_collision
        self.collision_fun[(PED,VEH)] = PED_VEH_collision
        self.collision_fun[(VEH,PED)] = VEH_PED_collision
        self.collision_fun[(VEH,VEH)] = VEH_VEH_collision


        self.max_Nnode = algo_config.max_clique_size
        self.dynamic[PED] = dynamics.DoubleIntegrator(
                "ped_dynamics",
                abound = algo_config.dynamics.pedestrain.axy_bound,
                vbound = [-algo_config.dynamics.pedestrain.max_speed,algo_config.dynamics.pedestrain.max_speed])

        self.dynamic[VEH] = dynamics.Unicycle(
                "veh_dynamics",
                max_steer=algo_config.dynamics.vehicle.max_steer,
                max_yawvel=algo_config.dynamics.vehicle.max_yawvel,
                acce_bound=algo_config.dynamics.vehicle.acce_bound
            )


        if "perturb" in algo_config and algo_config.perturb.enabled:
            self.N_pert = algo_config.perturb.N_pert
            theta = algo_config.perturb.OU.theta
            sigma = algo_config.perturb.OU.sigma
            scale = torch.tensor(algo_config.perturb.OU.scale)
            if self.dynamic[VEH] is not None:
                assert scale.shape[0]==self.dynamic[VEH].udim
                self.pert = DynOrnsteinUhlenbeckPerturbation(theta*torch.ones(self.dynamic[VEH].udim),sigma*scale,self.dynamic[VEH])
            else:
                assert scale.shape[0]==3
                self.pert = OrnsteinUhlenbeckPerturbation(theta*torch.ones(3),sigma*scale)
        else:
            self.N_pert=0
            self.pert=None

        # self.pred_state_length = dict()
        # for node_type in self.node_types:
        #     self.pred_state_length[node_type] = int(
        #         np.sum(
        #             [
        #                 len(entity_dims)
        #                 for entity_dims in self.pred_state[node_type].values()
        #             ]
        #         )
        #     )

        self.create_graphical_model()

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self,name,model):
        self.node_modules[name] = model

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):

        #####################
        #   Edge Encoders   #
        #####################
        edge_encoder_input_size = self.algo_config.edge_encoding_dim
        
        for edge_type in self.edge_types:

            model = getattr(
                ModelUtils,
                self.algo_config.edge_pre_enc_net[edge_type[0]][edge_type[1]],
            )
            self.add_submodule(
                str(edge_type[0]) + "->" + str(edge_type[1]) + "/edge_pre_encoding",
                model=model(edge_encoder_input_size),
            )
            self.edge_pre_enc_net[edge_type] = self.node_modules[
                str(edge_type[0]) + "->" + str(edge_type[1]) + "/edge_pre_encoding"
            ]

            self.add_submodule(
                str(edge_type[0]) + "->" + str(edge_type[1]) + "/edge_encoder",
                model=nn.LSTM(
                    input_size=edge_encoder_input_size,
                    hidden_size=self.algo_config.enc_rnn_dim_edge,
                    batch_first=True,
                ),
            )

        ############################
        #   Node History Encoder   #
        ############################
        node_enc_dim = self.algo_config.node_encoding_dim
        for node_type in self.node_types:
            model = getattr(
                ModelUtils,
                self.algo_config.node_pre_encode_net[node_type],
            )

            self.add_submodule(
                node_type + "/node_pre_encoder",
                model=model(
                    node_enc_dim, use_lane_info=False
                ),
            )

            self.add_submodule(
                node_type + "/node_history_encoder",
                model=nn.LSTM(
                    input_size=node_enc_dim,
                    hidden_size=self.algo_config.enc_rnn_dim_history,
                    batch_first=True,
                ),
            )

        ###########################
        #   Node Future Encoder   #
        ###########################
        for node_type in self.node_types:
            self.add_submodule(
                node_type + "/node_future_encoder",
                model=nn.LSTM(
                    input_size=node_enc_dim,
                    hidden_size=self.algo_config.enc_rnn_dim_future,
                    bidirectional=True,
                    batch_first=True,
                ),
            )

            # These are related to how you initialize states for the node future encoder.


        ###################
        #   Map Encoder   #
        ###################
        if self.algo_config.use_map_encoding:
            for node_type in self.node_types:
                map_encoder = RasterizedMapEncoder(
                    model_arch=self.algo_config.map_encoder.model_arch,
                    input_image_shape=self.modality_shapes["static"],
                    feature_dim=self.algo_config.map_feature_dim,
                    use_spatial_softmax=self.algo_config.spatial_softmax.enabled,
                    spatial_softmax_kwargs=self.algo_config.spatial_softmax.kwargs,
                    output_activation=nn.ReLU
                )
                self.add_submodule(
                    node_type + "/map_encoder",
                    model=map_encoder,
                )

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################

        z_size = self.algo_config.latent_dim

        ################
        #### p_z_x #####
        ################
        state_enc_dim = dict()
        for nt in self.node_types:
            if (
                self.algo_config.use_map_encoding
            ):
                state_enc_dim[nt] = (
                    self.algo_config.enc_rnn_dim_history
                    + self.algo_config.map_feature_dim
                )
            else:
                state_enc_dim[nt] = self.algo_config.enc_rnn_dim_history

        edge_encoding_dim = {
            et: self.algo_config.enc_rnn_dim_edge for et in self.edge_types
        }
        # Gibbs distribution for joint latent distribution
        self.add_submodule(
            "p_z_x",
            model=clique_gibbs_distr(
                state_enc_dim=state_enc_dim,
                edge_encoding_dim=edge_encoding_dim,
                z_dim=z_size,
                edge_types=self.edge_types,
                node_types=self.node_types,
                node_hidden_dim=[64, 64],
                edge_hidden_dim=[64, 64],
            ),
        )

        ################
        #### q_z_xy ####
        ################
        # Gibbs distribution for joint latent distribution
        for node_type in self.node_types:
            state_enc_dim[node_type] += self.algo_config.enc_rnn_dim_future * 4
        self.add_submodule(
            "q_z_xy",
            model=clique_gibbs_distr(
                state_enc_dim=state_enc_dim,
                edge_encoding_dim=edge_encoding_dim,
                z_dim=z_size,
                edge_types=self.edge_types,
                node_types=self.node_types,
                node_hidden_dim=[64, 64],
                edge_hidden_dim=[64, 64],
            ),
        )

        #################################
        ##  policy network as decoder  ##
        #################################
        map_enc_dim = self.algo_config.map_feature_dim
        if self.algo_config.use_map_encoding:
            map_enc_dims = {nt:map_enc_dim for nt in self.node_types}

        else:
            map_enc_dims = None

        self.add_submodule(
            "policy_net",
            model=guided_policy_net(
                node_types=self.node_types,
                edge_types=self.edge_types,
                z_dim=z_size,
                rel_state_fun=self.rel_state_fun,
                collision_fun=self.collision_fun,
                dyn_net=self.dynamic,
                edge_encoding_net=self.edge_pre_enc_net,
                edge_enc_dim=self.algo_config["edge_encoding_dim"],
                map_enc_dim=map_enc_dims,
                history_enc_dim=self.algo_config["enc_rnn_dim_history"],
                obs_lstm_hidden_dim=self.algo_config["policy_obs_LSTM_hidden_dim"],
                guide_RNN_hidden_dim=self.algo_config["dec_rnn_dim"],
                FC_hidden_dim=self.algo_config["policy_FC_hidden_dim"],
                max_Nnode=self.max_Nnode,
                dt=self.dt,
                output_var = self.output_var,
            ),
        )

    def create_graphical_model(self):
        """
        Creates or queries all trainable components.

        """
        self.clear_submodules()

        self.create_node_models()




    def obtain_encoded_tensors(
        self,
        hist_state,
        fut_state,
        hist_state_local, 
        fut_state_local, 
        hist_mask,
        fut_mask,
        maps,
        extents,
        node_types,
    ):

        ##################
        # Encode History #
        ##################
        hist_enc = self.encode_node_history(hist_state_local,hist_mask,node_types)

        ##################
        # Encode Future #
        ##################
        if fut_state is not None:
            fut_enc = self.encode_node_future(fut_state_local,fut_mask,node_types)

        else:
            node_future_encoded = None

        ##############################
        # Encode Node Edges per Type #
        ##############################
        edge_enc = self.encode_edge(hist_state, hist_mask, extents, node_types)

        ################
        # Map Encoding #
        ################
        bs, Na, nchannel = maps.shape[:3]
        map_enc = list()
        if self.algo_config.use_map_encoding:
            for nt in self.node_types:
                nt_mask = hist_mask.any(-1)*(node_types==nt.value)
                map_enc.append(self.node_modules[nt + "/map_encoder"](TensorUtils.join_dimensions(maps,0,2)).reshape(bs,Na,-1)*nt_mask.unsqueeze(-1))

        map_enc = sum(map_enc)
        return hist_enc, fut_enc, edge_enc, map_enc

    def obtain_clique(self,hist_state,fut_state,hist_mask,fut_mask,node_types):
        bs,Na,Th = hist_state.shape[:3]
        device = hist_state.device
        Tf = fut_state.shape[-2]
        agent_avail = hist_mask.any(-1)
        edge_avail = agent_avail.unsqueeze(2)*agent_avail.unsqueeze(1)
        if self.algo_config.use_proj_dis:
            traj = torch.zeros_like(fut_state)
            zero_action = torch.zeros([bs,Na,Tf,2]).to(device)
            for nt in self.node_types:
                traj_nt,_,_ = self.dynamic[nt].forward_dynamics(hist_state[:,:,-1],zero_action,self.dt)
                traj+=traj_nt*(node_types==nt.value)[...,None,None]
        else:
            traj = hist_state
        
        pos = traj[...,:2]
        dis_mat = torch.linalg.norm(pos.unsqueeze(1)-pos.unsqueeze(2),dim=-1).min(-1)[0]
        dis_mat[~edge_avail]=np.inf
        edge_type = torch.stack([node_types.unsqueeze(1).repeat_interleave(Na,1),node_types.unsqueeze(2).repeat_interleave(Na,2)],-1)
        radius = torch.zeros([bs,Na,Na],device=device)
        for et in self.edge_types:
            radius_et = self.algo_config.adj_radius[et[0].name][et[1].name]
            flag = (edge_type==torch.tensor([et[0].value,et[1].value],device=device)).all(-1)
            radius[flag] = radius_et

        adj_mat = TensorUtils.to_numpy(radius/dis_mat)
        adj_mat[adj_mat<1]=0
        adj_mat[adj_mat==np.inf]=10
        clique_index = -1*np.ones([bs,Na],dtype=int)
        agent_avail_np = TensorUtils.to_numpy(agent_avail)
        for i in range(bs):
            adj_mat_i = adj_mat[i][np.ix_(agent_avail_np[i],agent_avail_np[i])]
            if self.max_Nnode is None:
                
                n_components, labels = connected_components(
                    csgraph=csr_matrix(adj_mat_i), directed=False, return_labels=True
                )
            else:
                n_components, labels = ModelUtils.break_graph_recur(adj_mat_i, self.max_Nnode)


            clique_index[i,agent_avail_np[i]] = labels
        assert self.max_Nnode is not None
        clique_index = TensorUtils.to_torch(clique_index,device=device)
        
        N_clique = int(clique_index.max()+1)
        clique_node_index = -1*torch.ones([bs,clique_index.max()+1,self.max_Nnode],device=device,dtype=torch.long)
        clique_order_index = -1*torch.ones_like(clique_index)
        for i in range(N_clique):
            count = torch.cumsum((clique_index==i).type(torch.int),1)-1
            clique_order_index[clique_index==i] = count[clique_index==i]
        
        for i in range(bs):
            for k in range(self.max_Nnode):
                
                
                nodes_ik = torch.where(clique_order_index[i]==k)[0]
                clique_index_ik = clique_index[i][nodes_ik]
                clique_node_index[i,clique_index_ik,k]=nodes_ik

        return clique_index, clique_node_index, clique_order_index


    def encode_node_history(self, hist_state, hist_mask ,node_types):
        """
        Encodes the nodes history.

        """
        hist_enc = list()
        bs,Na,Th = hist_state.shape[:3]
        for nt in self.node_types:
            
            mask_nt = hist_mask*(node_types==nt.value).unsqueeze(-1)
            hist_pre = self.node_modules[nt + "/node_pre_encoder"](hist_state)*mask_nt.unsqueeze(-1)
            lstm_out, _ = self.node_modules[nt + "/node_history_encoder"](TensorUtils.join_dimensions(hist_pre,0,2))
            hist_enc.append(lstm_out.reshape(bs,Na,Th,-1)*mask_nt.unsqueeze(-1))

        return sum(hist_enc)


    def encode_edge(self, hist_state, hist_mask, extents, node_types):
        """
        Encode edges history
        """
        device = hist_state.device
        bs,Na,Th = hist_mask.shape[:3]
        edge_type = torch.stack([node_types.unsqueeze(1).repeat_interleave(Na,1),node_types.unsqueeze(2).repeat_interleave(Na,2)],-1)
        edge_mask = (hist_mask.unsqueeze(1).repeat_interleave(Na,1))*(hist_mask.unsqueeze(2).repeat_interleave(Na,2))
        edge_state_0 = hist_state.unsqueeze(1).repeat_interleave(Na,1)
        edge_state_1 = hist_state.unsqueeze(2).repeat_interleave(Na,2)
        extents = extents.unsqueeze(-2).repeat_interleave(Th,-2)
        node_size_0 = extents.unsqueeze(1).repeat_interleave(Na,1)
        node_size_1 = extents.unsqueeze(2).repeat_interleave(Na,2)
        edge_enc = list()
        for et in self.edge_types:
            et_value = torch.tensor([et[0].value,et[1].value]).to(device)
            pre_enc_net = self.node_modules[str(et[0]) + "->" + str(et[1]) + "/edge_pre_encoding"]
            edge_mask_et = edge_mask*((edge_type==et_value).all(-1,keepdim=True))
            edge_mask_et *=(~torch.eye(Na,dtype=torch.bool,device=device)[None,:,:,None])
            edge_pre_encode = pre_enc_net(edge_state_0,edge_state_1,node_size_0,node_size_1)*edge_mask_et.unsqueeze(-1)
            lstm_model = self.node_modules[str(et[0]) + "->" + str(et[1]) + "/edge_encoder"]
            lstm_out,_ = lstm_model(TensorUtils.join_dimensions(edge_pre_encode,0,3))
            lstm_out = lstm_out.reshape(bs,Na,Na,Th,-1)
            edge_enc.append(edge_mask_et.unsqueeze(-1)*lstm_out)

        return sum(edge_enc)





    def encode_node_future(self,fut_state,fut_mask,node_types):
        """
        Encodes the nodes history.

        """
        fut_enc = list()
        bs,Na,Tf = fut_state.shape[:3]
        for nt in self.node_types:
            mask_nt = fut_mask*(node_types==nt.value).unsqueeze(-1)
            fut_pre = self.node_modules[nt + "/node_pre_encoder"](fut_state)*mask_nt.unsqueeze(-1)
            _, state = self.node_modules[nt + "/node_future_encoder"](TensorUtils.join_dimensions(fut_pre,0,2))
            lstm_out = ModelUtils.unpack_RNN_state(state)
            fut_enc.append(lstm_out.reshape(bs,Na,-1)*mask_nt.any(-1).unsqueeze(-1))

        return sum(fut_enc)

    def encoder(self, node_types,hist_enc,fut_enc,edge_enc,map_enc,clique_index, clique_node_index, clique_order_index,num_samples=None,predict=False):
        """
        Encoder of the CVAE.

        """
        if map_enc is not None:
            node_enc = torch.cat((hist_enc[...,-1,:],map_enc),-1)
        else:
            node_enc = hist_enc
        
        log_pis_p, z = self.node_modules["p_z_x"](node_types,node_enc,edge_enc,clique_index, clique_node_index, clique_order_index)
        bs,Nc = log_pis_p.shape[:2]
        z = TensorUtils.join_dimensions(z,0,self.max_Nnode).reshape(1,1,-1,self.max_Nnode).repeat(bs,Nc,1,1)
        if fut_enc is not None and not predict:
            node_enc = torch.cat((node_enc,fut_enc),-1)
            log_pis_q, _ = self.node_modules["q_z_xy"](node_types,node_enc,edge_enc,clique_index, clique_node_index, clique_order_index)

        # joint latent cardinality for each node

        Na = node_types.shape[1]
        device = node_types.device
        
        
        logp = TensorUtils.join_dimensions(log_pis_p,2,2+self.max_Nnode)
        if "log_pi_clamp" in self.algo_config:
            logp = torch.clamp(logp, min=self.algo_config["log_pi_clamp"])
        p = torch.exp(logp)
        # dealing with cliques with size smaller than max_Nnode
        repeat_flag = torch.ones([bs,Nc,*(self.max_Nnode*[self.z_dim])],dtype=torch.bool,device=p.device)
        repeat_flag = TensorUtils.join_dimensions(repeat_flag,0,2)
        clique_node_index_tiled = TensorUtils.join_dimensions(clique_node_index,0,2)
        clique_order_index_tiled = TensorUtils.join_dimensions(clique_order_index,0,2)
        clique_index_tiled = TensorUtils.join_dimensions(clique_index,0,2)
        one_after = torch.arange(1,self.z_dim,dtype=torch.long,device=p.device)
        for i in range(self.max_Nnode):
            idx = torch.where(clique_node_index_tiled[:,i]==-1)[0]
            
            if idx.nelement()>0:
                repeat_flag[idx]=repeat_flag[idx].index_fill_(i+1,one_after,0)
        # leave only one entry of the repeated probabilities due the non-max cliques nonzero and set the rest to 0
        p = p*repeat_flag.reshape(bs,Nc,-1)*(logp!=0).any(-1).unsqueeze(-1)
        empty_flag = (p==0).all(-1)
        p[...,0]+=empty_flag
        p = p/torch.sum(p,-1,keepdim=True)
        if predict:
            if num_samples is None:
                num_samples = self.algo_config["pred_num_samples"]
            prob_list,idx_sel = torch.topk(p,num_samples,-1)
            z_list = torch.gather(z,2,idx_sel.unsqueeze(-1).repeat_interleave(self.max_Nnode,-1))
            dist = None
            
        else:
            nsample_greedy = self.algo_config["max_greedy_sample"]
            nsample_random = self.algo_config["max_random_sample"]
            
            # calculate q
            logq = TensorUtils.join_dimensions(log_pis_q,2,2+self.max_Nnode)

            if "log_pi_clamp" in self.algo_config:
                logq = torch.clamp(logq, min=self.algo_config["log_pi_clamp"])
            q = torch.exp(logq)
            # dealing with cliques with size smaller than max_Nnode

            # leave only one entry of the repeated probabilities due the non-max cliques nonzero and set the rest to 0
            q = q*repeat_flag.reshape(bs,Nc,-1)*(logq!=0).any(-1).unsqueeze(-1)
            
            empty_flag = (q==0).all(-1)
            q[...,0]+=empty_flag
            q = q/torch.sum(q,-1,keepdim=True)
            idx = q.argsort(-1,descending=True)
            idx_greedy = idx[...,:nsample_greedy]
            idx_random_in = torch.randint(nsample_greedy,p.shape[-1],[bs,Nc,nsample_random],device=p.device)
            idx_random = torch.gather(idx,2,idx_random_in)

            idx_sel = torch.cat((idx_greedy,idx_random),-1)
            
            z_list = torch.gather(z,2,idx_sel.unsqueeze(-1).repeat_interleave(self.max_Nnode,-1))
            prob_list = torch.gather(q,2,idx_sel)
            dist = (p,q,repeat_flag)
        
        numMode = idx_sel.shape[-1]
        prob_extended = torch.cat((prob_list,torch.zeros_like(prob_list[:,0:1])),1)
        clique_index_extend = clique_index.clone()
        clique_index_extend[clique_index_extend==-1]=Nc
        prob_node = torch.gather(prob_extended,1,clique_index_extend[...,None].repeat_interleave(numMode,-1))
        z_node = torch.zeros([bs*Na,numMode],dtype=torch.long).to(device)
        z_list_tiled = TensorUtils.join_dimensions(z_list,0,2)
        for i in range(self.max_Nnode):
            node_idx = torch.where(clique_order_index_tiled==i)[0]
            batch_idx = torch.where(clique_order_index==i)[0]
            clique_idx = clique_index_tiled[node_idx]
            clique_idx+=batch_idx*Nc

            z_node[node_idx,:] = z_list_tiled[clique_idx,:,i]
        z_node = z_node.reshape(bs,Na,numMode)
        
        return z_list,z_node, prob_list, prob_node, dist
            
   

    def decoder(
        self,
        state_history,
        state_history_local,
        node_history_encoded,
        encoded_map,
        node_size,
        lane_st,
        clique_index,
        node_types,
        z_list,
        z_node,
        Tf,
        cond=None,
    ):
        """
        Decoder of the CVAE.

        """

        model = self.node_modules["policy_net"]
        # turn the latent to one-hot
        (
            state_pred,
            state_pred_local,
            input_pred,
            des_traj,
            var,
            tracking_error,
        ) = model(
            state_history,
            state_history_local,
            node_history_encoded,
            encoded_map,
            node_size, 
            lane_st, 
            Tf, 
            z_list,
            z_node,
            clique_index, 
            node_types,
            cond
        )


        return (
                state_pred,
                state_pred_local,
                input_pred,
                des_traj,
                var,
                tracking_error,
        )
    def sample(self,batch,n):
        return self.forward(batch,num_samples=n)

    def forward(self, batch,cond_traj=None,num_samples=None):
        hist_pos = batch["history_positions"]
        hist_yaw = batch["history_yaws"]
        fut_pos = batch["target_positions"]
        fut_yaw = batch["target_yaws"]
        fut_mask = batch["target_availabilities"]
        hist_mask = batch["history_availabilities"]
        maps = batch["image"]
        extents = batch["extent"][...,:2]
        
        node_types = batch["agent_type"]

        hist_pos_local = batch_nd_transform_points(hist_pos,batch["agents_from_center"].unsqueeze(2))
        hist_yaw_local = hist_yaw-hist_yaw[:,:,-1:]
        fut_pos_local = batch_nd_transform_points(fut_pos,batch["agents_from_center"].unsqueeze(2))
        fut_yaw_local = fut_yaw-hist_yaw[:,:,-1:]
        device = hist_pos.device
        bs,Na,Th = hist_pos.shape[:3]
        Tf = fut_pos.shape[-2]
        hist_state = torch.zeros([bs,Na,Th,4]).to(device)
        fut_state = torch.zeros([bs,Na,Tf,4]).to(device)
        hist_state_local = torch.zeros([bs,Na,Th,4]).to(device)
        fut_state_local = torch.zeros([bs,Na,Tf,4]).to(device)
        for nt in self.node_types:
            hist_state+=self.dynamic[nt].get_state(hist_pos,hist_yaw,self.dt,hist_mask)*hist_mask.unsqueeze(-1)*(node_types==nt.value).view(bs,Na,1,1)
            fut_state+=self.dynamic[nt].get_state(fut_pos,fut_yaw,self.dt,fut_mask)*fut_mask.unsqueeze(-1)*(node_types==nt.value).view(bs,Na,1,1)
            hist_state_local+=self.dynamic[nt].get_state(hist_pos_local,hist_yaw_local,self.dt,hist_mask)*hist_mask.unsqueeze(-1)*(node_types==nt.value).view(bs,Na,1,1)
            fut_state_local+=self.dynamic[nt].get_state(fut_pos_local,fut_yaw_local,self.dt,fut_mask)*fut_mask.unsqueeze(-1)*(node_types==nt.value).view(bs,Na,1,1)
        clique_index, clique_node_index, clique_order_index = self.obtain_clique(hist_state,fut_state,hist_mask,fut_mask,node_types)
        
        hist_enc,fut_enc,edge_enc,map_enc = self.obtain_encoded_tensors(hist_state, fut_state, hist_state_local, fut_state_local, hist_mask, fut_mask, maps, extents, node_types)
        
        z_list,z_node, prob, prob_node, dist = self.encoder(node_types,hist_enc,fut_enc,edge_enc,map_enc,clique_index, clique_node_index, clique_order_index,num_samples=num_samples)
        if self.ego_conditioning:
            if cond_traj is None:
                cond_state = torch.zeros_like(fut_state)
                cond_state[:,0]=fut_state[:,0]
                cond_state_local = torch.zeros_like(fut_state)
                cond_state_local[:,0]=fut_state_local[:,0]
                cond_state = [cond_state.unsqueeze(1)]
                cond_state_local = [cond_state_local.unsqueeze(1)]
                M = 1
                if self.N_pert>0:
                    ego_traj_tiled = fut_state[:,0].repeat_interleave(self.N_pert,0)
                    pert_res = self.pert.perturb(dict(target_positions=ego_traj_tiled[...,:2],target_yaws=ego_traj_tiled[...,3:],\
                                                                target_availabilities=torch.ones([bs*self.N_pert,Tf],device=device,dtype=torch.bool),step_time=self.dt))

                    ego_traj_tiled_pert = torch.cat((pert_res["target_positions"],ego_traj_tiled[...,2:3],pert_res["target_yaws"]),-1)
                    cond_state_pert = torch.zeros([bs,self.N_pert,Na,Tf,fut_state.shape[-1]],device=device)
                    cond_state_pert[:,:,0] = ego_traj_tiled_pert.reshape(bs,self.N_pert,Tf,fut_state.shape[-1])
                    pert_ego_pos_local = batch_nd_transform_points(pert_res["target_positions"],batch["agents_from_center"][:,0,None].repeat_interleave(self.N_pert,0))
                    pert_ego_yaw_local = pert_res["target_yaws"]-hist_yaw[:,0,-1,None].repeat_interleave(self.N_pert,0)
                    ego_traj_tiled_pert_local = torch.cat((pert_ego_pos_local,ego_traj_tiled[...,2:3],pert_ego_yaw_local),-1)
                    cond_state_pert_local = torch.zeros([bs,self.N_pert,Na,Tf,fut_state.shape[-1]],device=device)
                    cond_state_pert_local[:,0]=ego_traj_tiled_pert_local.reshape(bs,self.N_pert,Tf,fut_state.shape[-1])
                    cond_state.append(cond_state_pert)
                    cond_state_local.append(cond_state_pert_local)
                    M += self.N_pert
                if self.UAC:
                    dummy_cond_state = torch.zeros([bs,1,Na,Tf,fut_state.shape[-1]],device=device)
                    cond_state.insert(0,dummy_cond_state)
                    cond_state_local.insert(0,dummy_cond_state)
                    M += 1
                cond_state = torch.cat(cond_state,1)
                cond_state_local = torch.cat(cond_state_local,1)
                cond = (TensorUtils.join_dimensions(cond_state,0,2),TensorUtils.join_dimensions(cond_state_local,0,2))
                cond_traj = cond_state[...,[0,1,3]]
            else:
                M=1
                raise NotImplementedError
        else:
            M=1
            cond=None
            cond_traj = None

                
        if M==1:
            (
                state_pred,
                state_pred_local,
                input_pred,
                des_traj,
                var,
                tracking_error
            ) = self.decoder(
                        hist_state,
                        hist_state_local,
                        hist_enc,
                        map_enc,
                        extents,
                        None,
                        clique_index,
                        node_types,
                        z_list,
                        z_node,
                        Tf,
                        cond,
            )
        else:
            (
                state_pred,
                state_pred_local,
                input_pred,
                des_traj,
                var,
                tracking_error
            ) = self.decoder(
                        hist_state.repeat_interleave(M,0),
                        hist_state_local.repeat_interleave(M,0),
                        hist_enc.repeat_interleave(M,0),
                        map_enc.repeat_interleave(M,0),
                        extents.repeat_interleave(M,0),
                        None,
                        clique_index.repeat_interleave(M,0),
                        node_types.repeat_interleave(M,0),
                        z_list.repeat_interleave(M,0),
                        z_node.repeat_interleave(M,0),
                        Tf,
                        cond,
            )
        edge_mask = (clique_index.unsqueeze(1)==clique_index.unsqueeze(2))*(clique_index.unsqueeze(1)>=0)
        res = [state_pred,state_pred_local,input_pred,des_traj]
        if var is not None:
            res.append(var)
        res = TensorUtils.reshape_dimensions(res,0,1,[bs,M])
        res = TensorUtils.recursive_dict_list_tuple_apply(res,{torch.Tensor: lambda x:x.transpose(2,3),type(None): lambda x: x,})
        if var is not None:
            state_pred, state_pred_local, input_pred, des_traj, var = res
        else:
            state_pred, state_pred_local, input_pred, des_traj = res
        agent_avail = batch["target_availabilities"].any(-1)
        pred_batch = dict(
            trajectories=state_pred[...,[0,1,3]],
            p=prob_node.unsqueeze(1).repeat_interleave(M,1),
            z_node = z_node,
            z_list = z_list,
            des_traj=des_traj,
            controls=input_pred,
            agent_avail=agent_avail,
            cond_traj=cond_traj,
            var = var,
            edge_mask = edge_mask,
            clique_index = clique_index,
        )
        if dist is not None:
            pred_batch["dist"] = dict(p=dist[0],q=dist[1],repeat_flag=dist[2])

        pred_batch.update(self._traj_to_preds(pred_batch["trajectories"]))
        return pred_batch
    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
    


    def compute_losses(self,pred_batch,data_batch):

        availability = data_batch["target_availabilities"]
        extent = data_batch["extent"][...,:2]
        raw_type = data_batch["type"]
        bs,M,numMode,Na,Tf = pred_batch["trajectories"].shape[:5]
        
        target_traj = torch.cat((data_batch["target_positions"],data_batch["target_yaws"]),-1)
        target_traj = target_traj[...,:Tf,:]
        
        availability = availability[...,:Tf]
        traj_pred = pred_batch["trajectories"]
        traj_pred_tiled = TensorUtils.join_dimensions(traj_pred,0,3)
        prob = pred_batch["p"]

        if self.UAC:
            EC_pred = traj_pred[:,1:]
            Uncond_pred = traj_pred[:,0]
            cond_traj = TensorUtils.join_dimensions(pred_batch["cond_traj"][:,1:,...],0,2)
            var = pred_batch["var"][:,1:]
            pred_loss, goal_loss = MultiModal_trajectory_loss(
                predictions=Uncond_pred, 
                targets=target_traj, 
                availabilities=availability, 
                prob=prob[:,0], 
                weights_scaling=self.weights_scaling,
                calc_goal_reach=False,
                gamma = self.gamma
            )
            flat_pred = EC_pred[:,:,:,1:,...,:2].reshape(-1,2)

            dev_loss = NLL_GMM_loss(flat_pred, 
                                    Uncond_pred[:,:,1:,...,:2].repeat_interleave(M-1,0).reshape(-1,1,2),
                                    var[:,:,:,1:].reshape(-1,1,2), 
                                    torch.ones([flat_pred.shape[0],1],device=var.device),
                                    detach=False, 
                                    mode="sum")
            dev_loss = torch.clip(dev_loss,min = 0)
            div_score = diversity_score(TensorUtils.join_dimensions(EC_pred[...,:2],0,2),availability.repeat_interleave(M-1,0))/(M-1)

        else:
            pred_loss, goal_loss = MultiModal_trajectory_loss(
                predictions=TensorUtils.join_dimensions(traj_pred,0,2), 
                targets=target_traj.repeat_interleave(M,0), 
                availabilities=availability.repeat_interleave(M,0), 
                prob=TensorUtils.join_dimensions(prob,0,2), 
                weights_scaling=self.weights_scaling,
                calc_goal_reach=False,
                gamma = self.gamma
            )
            EC_pred = TensorUtils.join_dimensions(traj_pred,0,2)
            cond_traj = TensorUtils.join_dimensions(pred_batch["cond_traj"],0,2)
            dev_loss = 0.0
            div_score = diversity_score(EC_pred[...,:2],availability.repeat_interleave(M,0))/M
        

        
        # compute collision loss
        
        pred_edges, type_mask = batch_utils().gen_edges_masked(
            raw_type.repeat_interleave(M*numMode,0),
            extent.repeat_interleave(M*numMode,0),
            traj_pred_tiled,
        )
        edge_mask = pred_batch["edge_mask"].repeat_interleave(M*numMode,0)
        type_mask = {k:v*edge_mask for k,v in type_mask.items()}
        
        coll_loss = collision_loss_masked(pred_edges,type_mask)/bs
        if not isinstance(coll_loss,torch.Tensor):
            coll_loss = torch.tensor(coll_loss).to(extent.device)
        kl_loss = KLD_discrete_with_zero(pred_batch["dist"]["p"],pred_batch["dist"]["q"],logmin=self.algo_config["log_pi_clamp"]).sum(-1).mean()
        
        losses = OrderedDict(
            prediction_loss=pred_loss,
            kl_loss = kl_loss,
            collision_loss=coll_loss,
            diversity_loss = -div_score,
            yaw_reg_loss = torch.mean(pred_batch["controls"][..., 1] ** 2 ),
            deviation_loss = dev_loss
        )
        if self.algo_config.goal_conditional:
            losses["goal_loss"] = goal_loss
        return losses