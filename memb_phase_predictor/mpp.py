# Author: Daniel Pastor Ramirez Echemendia

# import libraries
import pathlib
import pickle
import random
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import MDAnalysis as mda
import hmmlearn.hmm as hmm
from sklearn.preprocessing import LabelEncoder
from MDAnalysis.analysis.distances import distance_array
from lipyphilic.lib.assign_leaflets import AssignLeaflets
from lipyphilic.transformations import triclinic_to_orthorhombic
from lipyphilic.lib.order_parameter import SCC


class Predictor():
    def __init__(self, structure, trajectory, begin=-1000, skip=25, b_type='triclinic'):
        self.universe = mda.Universe(structure, trajectory)
        if b_type == 'triclinic':
            self.universe.add_transformations(triclinic_to_orthorhombic(ag=self.universe.select_atoms('resname DPPC DOPC')))
        self.nframes = self.universe.trajectory.n_frames
        self.leaflets = None
        self.lip_state_lower = None
        self.lip_state_upper = None
        self.scc = None
        self.N = 5
        self.begin = begin
        self.skip = skip
    

    def itentify_membrane_leaflets(self, glycerol_chain = 'name GL1 GL2', n_bins=10, save=True):
        print('\nAssigning lipids to leaflets....')
        # atom selection covering all lipids in the bilayer
        self.leaflets = AssignLeaflets(universe=self.universe, lipid_sel=glycerol_chain, n_bins=n_bins)
        self.leaflets.run(start=None, stop=None, step=None, verbose=True)
        self.lip_state_lower = np.full((np.count_nonzero(self.leaflets.leaflets[:,-1] == -1), len(self.universe.trajectory[self.begin::self.skip])), '11')
        self.lip_state_upper = np.full((np.count_nonzero(self.leaflets.leaflets[:,-1] == 1), len(self.universe.trajectory[self.begin::self.skip])), '11')
        if save:
            results_directory = pathlib.Path("results/leaflets")
            # Create the directory if it doesn't already exist
            results_directory.resolve().mkdir(exist_ok=True, parents=True)
            filename = results_directory.joinpath("leaflets.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(self.leaflets, f)


    def calculate_order_parameters(self, tail1 = 'name ??A', tail2 = 'name ??B', save=True):
        print('\nCalculating order parameters....')
        # tail1
        scc_sn1 = SCC(universe = self.universe, tail_sel = tail1)
        print('tail1....\n')
        scc_sn1.run(start=None, stop=None, step=None, verbose=True)
        # tail2
        scc_sn2 = SCC(universe = self.universe, tail_sel = tail2)
        print('tail2....\n')
        scc_sn2.run(start=None, stop=None, step=None, verbose=True)
        self.scc = SCC.weighted_average(scc_sn1, scc_sn2)
        if save:
            results_directory = pathlib.Path("results/order_parameters")
            # Create the directory if it doesn't already exist
            results_directory.resolve().mkdir(exist_ok=True, parents=True)
            filename = results_directory.joinpath("order_parameters.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(self.scc, f)


    def cal_xycom(self, leaflet_group):
        # function to calculate center of mass
        lip_com = []
        for i in range(0, len(leaflet_group.residues)):
            if i == 0:
                lip_com = leaflet_group.residues[i].atoms.center_of_mass()
            else:
                lip_com = np.concatenate((lip_com, leaflet_group.residues[i].atoms.center_of_mass()), axis = 0)
        lip_com = lip_com.reshape(len(leaflet_group.residues), 3)
        x_zeros = np.zeros(len(leaflet_group.residues))
        lip_com[:,2] = x_zeros
        return lip_com


    def lstate(self, leaflet_group, lip_com, box, Scd, N):
        '''function to check the nearest N lipids'''
        #cal the distance array of two groups
        distarr = distance_array(lip_com, lip_com, box)
        Scd = Scd.reshape(1, distarr.shape[0])
        #count lip no. based on the dist array
        lip_state = []
        for i in range(0, distarr.shape[0]):
            lip_distarr = distarr[i, :]
            lip_index_sort = np.argsort(lip_distarr)
            lip_index_nearest = lip_index_sort[0:N]
            dp_num = 0
            for j in range(0, len(lip_index_nearest)):
                if leaflet_group.residues[lip_index_nearest[j]].resname == 'DPPC':
                    dp_num +=1
                else:
                    pass
            if Scd[0, i] <= 0.33 and leaflet_group.residues[i].resname == 'DPPC':
                Scd_state = 0
            elif Scd[0, i] > 0.33 and leaflet_group.residues[i].resname == 'DPPC':
                Scd_state = 1
            elif Scd[0, i] <= 0.23 and leaflet_group.residues[i].resname == 'DOPC':
                Scd_state = 0
            elif Scd[0, i] > 0.23 and leaflet_group.residues[i].resname == 'DOPC':
                Scd_state = 1

            state_temp = str(dp_num)+str(Scd_state)
            lip_state.append(state_temp)
        return lip_state


    def determine_observations(self, save=True):
        print('\nDetermining observations....')
        idx_frame = 0
        for ts in tqdm(self.universe.trajectory[self.begin::self.skip]):
            ##cal dist array
            box = ts.dimensions

            #residues in lower and upper leaflets
            lower_leaflet_residues = self.universe.residues[self.leaflets.leaflets[:,-1] == -1]
            upper_leaflet_residues = self.universe.residues[self.leaflets.leaflets[:,-1] == 1]

            #cal com of lipids
            lip_com1 = self.cal_xycom(self.lower_leaflet_residues)
            lip_com2 = self.cal_xycom(self.upper_leaflet_residues)

            #print lip_res
            lip_com1 = lip_com1.astype(np.float32)
            lip_com2 = lip_com2.astype(np.float32)

            #masks for lower and upper leaflets
            lower_leaflet_mask = (self.leaflets.leaflets[:, -1] == -1)
            upper_leaflet_mask = (self.leaflets.leaflets[:, -1] == 1)

            lip_state1 = self.lstate(lower_leaflet_residues, lip_com1, box, self.scc.SCC[lower_leaflet_mask, ts.frame], self.N)
            lip_state2 = self.lstate(upper_leaflet_residues, lip_com2, box, self.scc.SCC[upper_leaflet_mask, ts.frame], self.N)

            for i in range(0, len(lip_state1)):
                self.lip_state_lower[i, idx_frame] = lip_state1[i]
            for i in range(0, len(lip_state2)):
                self.lip_state_upper[i, idx_frame] = lip_state2[i]
            idx_frame += 1

        if save:
            results_directory = pathlib.Path("results/HMM")
            results_directory.mkdir(exist_ok=True)
            filename = results_directory.joinpath('observations_lower.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(self.lip_state_lower, f)
            filename = results_directory.joinpath('observations_upper.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(self.lip_state_upper, f)


    def train_hmm(self, replicates, save=True):
        print('\nTraining HMM....')
        # train HMM
        state_both = np.concatenate((self.lip_state_upper, self.lip_state_lower), axis = 0).T
        state_both = state_both.astype(np.int8)

        #define observation states
        #the first No is the no. of dppc in the local
        #the second no is based on the order parameter, dp<0.33, do<0.23 is 0, otherwise 1
        # observations = ['00', '10', '20', '30', '40', '50', '60', '01', '11', '21', '31', '41', '51', '61']
        observations = ['00', '10', '20', '30', '40', '50', '01', '11', '21', '31', '41', '51']
        n_observations = len(np.unique(state_both))

        #build the model
        #define states
        states = ['phase1','phase2']
        n_states = len(states)

        for k in range(0, replicates):
            # This is our output array 
            lipid_order = np.zeros((len(self.universe.residues), len(self.universe.trajectory[self.begin::self.skip])), dtype=np.int8)
            #define start probabilities
            temp = random.random()
            start_probabilities_init = np.array([temp, 1-temp])
            #define transition probabilities
            temp = np.random.rand(2, 2)
            temp_sum = np.array([temp[0, :].sum(), temp[1, :].sum()]).reshape(2, 1)
            transition_probabilities_init = np.divide(temp, temp_sum)
            #define emission probabilites
            temp = np.random.rand(2, n_observations)
            temp_sum = np.array([temp[0, :].sum(), temp[1, :].sum()]).reshape(2, 1)
            emission_probabilities_init = np.divide(temp, temp_sum) 

            #define the model
            hmodel = hmm.MultinomialHMM(n_components = n_states, algorithm = 'viterbi', n_iter = 10000, tol = 0.00005)
            hmodel.startprob_ = start_probabilities_init
            hmodel.transmat_ = transition_probabilities_init
            hmodel.emismatprob_ = emission_probabilities_init
            
            state_transform = LabelEncoder().fit_transform(state_both.flatten('F'))
            #the states must cover all values from min(x) to max(x), have to be integer
            #e.g. [0, 1, 1, 2] is valid, but [0, 1, 1, 5] is not
            #LabelEncoder().fit_transform(array) transform the states to consecutive numbers
            #in order to fullfill the aforementioned requirements
            #another solution is to assign numbers based on the states manually
            length_seq = []
            for i in range(0, len(self.universe.residues)):
                length_seq.append(len(self.universe.trajectory[self.begin::self.skip]))
            
            state_seq = state_transform.reshape(state_transform.shape[0], 1).astype(np.int8)
            hmodel = hmodel.fit(state_seq, length_seq)
            
            logprob, state_predic = hmodel.decode(state_seq, length_seq, algorithm='viterbi')
            
            print('Is the HMM model for phase separation converged?:\t')
            print(hmodel.monitor_.converged)
            
            #reshape the state_predic array
            state_predic = state_predic.reshape(len(self.universe.residues), int(len(state_predic)/len(self.universe.residues))).T
            #dump the data   
            for i in range(0, state_predic.shape[0]):
                for j in range(0, state_predic.shape[1]):
                    if state_predic[i, j] == 0:
                        lipid_order[j, i] = -1
                    elif state_predic[i, j] == 1:
                        lipid_order[j, i] = 1

            # Save the ordered states of all lipids
            if save:
                results_directory = pathlib.Path("results/HMM")
                results_directory.mkdir(exist_ok=True)
                filename = results_directory.joinpath('lipid_order_' + str(k) + '.npy')
                np.save(filename, lipid_order)

    def analyze_hmm(self, replicates):
        print('\nAnalyzing HMM....')
        # Loading the lipid_order results (phase determination)
        results_directory = pathlib.Path("results/HMM")
        for k in range(0, replicates):
            lipid_order = np.load(results_directory.joinpath('lipid_order_' + str(k) + '.npy'))
            # Different composition between the phases
            df_content = pd.DataFrame({'Phase': [], 'DPPC': [], 'DOPC': []})
            #residues in lower and upper leaflets
            lower_leaflet_residues = self.universe.residues[self.leaflets.leaflets[:,-1] == -1]
            upper_leaflet_residues = self.universe.residues[self.leaflets.leaflets[:,-1] == 1]
            for phase in [-1, 1]:
                count_dppc = []
                count_dopc = []
                for i in range(lipid_order.shape[1]):
                    order_idxs = [i for i, x in enumerate((lipid_order == phase)[:,i]) if x]
                    dppc_no = 0
                    dopc_no = 0
                    for index in order_idxs:
                        if (index+1 in upper_leaflet_residues.residues[upper_leaflet_residues.residues.resnames == 'DPPC'].resids) or (index+1 in lower_leaflet_residues.residues[lower_leaflet_residues.residues.resnames == 'DPPC'].resids):
                            dppc_no += 1
                        if (index+1 in upper_leaflet_residues.residues[upper_leaflet_residues.residues.resnames == 'DOPC'].resids) or (index+1 in lower_leaflet_residues.residues[lower_leaflet_residues.residues.resnames == 'DOPC'].resids):
                            dopc_no += 1
                    count_dppc.append(dppc_no)
                    count_dopc.append(dopc_no)
                df_content = df_content.append({'Phase': phase, 'DPPC': np.mean(count_dppc), 'DOPC': np.mean(count_dopc)}, ignore_index=True)

            dppc_La = df_content.iat[0,1]/(df_content.iat[0,2] + df_content.iat[0,1])
            dopc_La = df_content.iat[0,2]/(df_content.iat[0,2] + df_content.iat[0,1])
            dppc_gel = df_content.iat[1,1]/(df_content.iat[1,1] + df_content.iat[1,2])
            dopc_gel = df_content.iat[1,2]/(df_content.iat[1,1] + df_content.iat[1,2])

            f = open(results_directory.joinpath('summary.txt'), 'w')
            f.write(str(round(dppc_gel, 3)) + ':' + str(round(dopc_gel, 3)) + '\t' + str(round(dppc_La, 3)) + ':' + str(round(dopc_La, 3)))
            f.write('\n')
            f.close()











