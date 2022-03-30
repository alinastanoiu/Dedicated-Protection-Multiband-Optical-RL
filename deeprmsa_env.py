import gym
import numpy as np
import time
import pandas as pd
from .rmsa_env import RMSAEnv
from .optical_network_env import OpticalNetworkEnv


class DeepRMSAEnv(RMSAEnv):

    def __init__(self, topology=None, j=1,
                 episode_length=1000,
                 mean_service_holding_time=25.0,
                 mean_service_inter_arrival_time=.1,
                 num_spectrum_resources= [34, 48, 76, 113], # should be  [344, 480, 760, 1136] 
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False):
        super().__init__(topology=topology,
                         episode_length=episode_length,
                         load=mean_service_holding_time / mean_service_inter_arrival_time,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths,
                         allow_rejection=allow_rejection,
                         reset=False)
        self.bands = 4
        self.j = j
        shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths * self.bands * 2 #multiply by bands and by 2 for backup path
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths*self.j*self.bands*2 + self.reject_action) 
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_counters=False)
        self.blockingReason = pd.DataFrame(columns=['Source', 'Destination', 'Reason', 'Path', 'Nodes'])
    def step(self, action: int):
        # print("k Paths: ".format(self.k_paths))
        if action < (2*self.k_paths*self.j*self.j*self.bands*self.bands - 2*self.bands*(self.k_paths -1)):  # action is for assigning a path, added bands
            w_path, w_block, w_band, b_path, b_block, b_band = self._get_path_block_id(action) #added band here and in function
            print("working path: {}, working block: {}, working band: {}, backup path: {}, backup block: {}, backup band: {} ".format(w_path, w_block, w_band, b_path, b_block, b_band))
            # self.band = band
            # print("k shortest paths: {}".format(self.k_shortest_paths))
            w_initial_indices, w_lengths = self.get_available_blocks(w_path, w_band)
            b_initial_indices, b_lengths = self.get_available_blocks(b_path, b_band)
            print("w_initial_indeces: {}, b_initial_indeces: {}".format(w_initial_indices, b_initial_indices))
            if (w_block < len(w_initial_indices)) and (b_block < len(b_initial_indices)):
                print("call rmsa step function")
                return super().step([w_path, w_band, w_initial_indices[w_block], b_path, b_band, b_initial_indices[b_block]])   ### CHECK whether call
            else:
                self.blockingReason = self.blockingReason.append(pd.DataFrame({'Source': [self.service.source], 'Destination':[self.service.destination], 'Reason': ['Slots Full']}), ignore_index = True)
                print("action wasn't valid")
                return super().step([self.k_paths, self.bands, self.num_spectrum_resources[w_band],
                                     self.k_paths, self.bands, self.num_spectrum_resources[b_band]])  # no connection
        else:
            self.blockingReason = self.blockingReason.append(pd.DataFrame({'Source': [self.service.source], 'Destination':[self.service.destination], 'Reason': ['Action out of range']}), ignore_index = True)
            print("action wasn't valid 1st if")
            return super().step([self.k_paths, self.bands, self.num_spectrum_resources[w_band],
                                 self.k_paths, self.bands, self.num_spectrum_resources[b_band]])    #no connection

    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(self.service.source_id, self.service.destination_id)
        max_node = max(self.service.source_id, self.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths * self.bands*2, 2 * self.j + 3), fill_value=-1.)    #added bands #multiplied by 2 for dedicated protection
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]): 
            #print("-------------------")
            #print(self.j)
            for idband in range(self.bands):        #including bands
                #print(idp)
                #print(self.k_paths * idband)
                available_slots = self.get_available_slots(path)   
                num_slots = self.get_number_slots(path, idband)     #updated for MB and modulation formats.
                initial_indices, lengths = self.get_available_blocks(idp, idband)

                for idz, (initial_index, length) in enumerate(zip(initial_indices, lengths)):              
                    # initial slot index                        
                    spectrum_obs[idp +(self.k_paths * idband), idz * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources[idband]) / self.num_spectrum_resources[idband] #spectrum res of band

                    # number of contiguous FS available
                    spectrum_obs[idp+ (self.k_paths * idband), idz * 2 + 1] = (length - 8) / 8
                spectrum_obs[idp+ (self.k_paths * idband), self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary

                idx, values, lengths = DeepRMSAEnv.rle(available_slots)

                av_indices = np.argwhere(values == 1) # getting indices which have value 1
                spectrum_obs[idp + (self.k_paths * idband), self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources[idband]) / self.num_spectrum_resources[idband] # total number available FSs
                spectrum_obs[idp + (self.k_paths * idband), self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of FS blocks available           #??? why k_paths * times
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.service.bit_rate / 100
        #print("Spectrum Obs", bit_rate_obs, source_destination_tau, spectrum_obs, num_slots)
        #time.sleep(0.05)
        return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
            .reshape(self.observation_space.shape)

    def reward(self):
        #print(self.service.accepted)
        return 1 if self.service.accepted else -1

    def reset(self, only_counters=True):
        return super().reset(only_counters=only_counters)

    def _get_path_block_id(self, action: int) -> (int, int, int, int, int, int):   #updated for MB
        print(action)
        print("k: {}, j: {}, b: {}".format(self.k_paths, self.j, self.bands))
        w_path = action % (self.j*self.j*self.bands*2*self.k_paths) // (self.j*self.bands*self.k_paths*self.j)          #floor division to give largest integer possible
        w_block = action % (self.j*self.j*self.k_paths*self.bands) // (self.j*self.k_paths*self.bands)
        w_band = action % self.bands
        b_path = action % (self.j*self.k_paths*self.bands) // (self.j* self.bands)
        b_block = action % (self.j*self.bands) // self.bands
        b_band = action % self.bands
        print(w_path, w_block, w_band, b_path, b_block, b_band)
        return w_path, w_block, w_band, b_path, b_block, b_band


#### functions below not used

def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, lengths = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        initial_indices, lengths = env.get_available_blocks(idp)
        if len(initial_indices) > 0: # if there are available slots
            return idp * env.j # this path uses the first one
    return env.k_paths * env.j
