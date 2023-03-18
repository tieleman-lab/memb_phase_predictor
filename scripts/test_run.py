import memb_phase_predictor as mpp
import os

os.chdir('/Users/danielramirez/Dropbox (Biocomputing)/GitHub/projects/memb_phase_predictor/memb_phase_predictor/data/')

# tt = mpp.Predictor('cg_good.gro', 'cg.xtc', b_type='orthorombic')
tt = mpp.Predictor('DPPC_DOPC_280K.gro', 'DPPC_DOPC_280K.xtc', b_type='triclinic')

tt.itentify_membrane_leaflets()
# tt.load_leaflet_data('results/leaflets/leaflets.pkl')
tt.calculate_order_parameters()
# tt.load_order_parameters('results/order_parameters/order_parameters.pkl')
tt.determine_observations()
# tt.load_observations('results/HMM/observations_lower.pkl', 'results/HMM/observations_upper.pkl')
tt.train_hmm(1)
tt.analyze_hmm(1)