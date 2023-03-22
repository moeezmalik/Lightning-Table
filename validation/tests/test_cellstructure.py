from cellstructure import Datasheet
import yaml

def test_one():

    elec_result_dict = {'eff': {'cols': [1], 'vals': ['22.400', '22.300', '22.200', '22.100', '22.000', '21.900', '21.800', '21.700', '21.600', '21.500', '21.400']}, 'isc': {'cols': [6], 'vals': ['9.970', '9.960', '9.950', '9.940', '9.930', '9.920', '9.910', '9.900', '9.890', '9.880', '9.870']}, 'voc': {'cols': [5], 'vals': ['0.689', '0.688', '0.687', '0.686', '0.685', '0.684', '0.683', '0.682', '0.681', '0.680', '0.679']}, 'impp': {'cols': [4], 'vals': ['9.550', '9.520', '9.500', '9.470', '9.440', '9.420', '9.390', '9.360', '9.340', '9.310', '9.280']}, 'vmpp': {'cols': [3], 'vals': ['0.591', '0.590', '0.589', '0.588', '0.587', '0.586', '0.585', '0.584', '0.583', '0.582', '0.581']}, 'pmpp': {'cols': [2], 'vals': ['5.640', '5.620', '5.590', '5.570', '5.540', '5.520', '5.490', '5.470', '5.440', '5.420', '5.390']}, 'ff': {'cols': None, 'vals': None}}
    temp_result_dict = {'isc': {'rows': [2], 'vals': ['+0.07%/k']}, 'voc': {'rows': [1], 'vals': ['-0.36%/k']}, 'pmpp': {'rows': [3], 'vals': ['-0.38%/k']}}

    with open("/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/patterns.yaml", "r") as stream:
        try:
            patterns = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    sample_ds = Datasheet(
        path_to_excel="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/tests/files/one.xlsx",
        path_to_clf="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/nb_classifier.pickle",
        path_to_vec="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/vectoriser.pickle"
    )
    
    sample_ds.extract_electrical_props(patterns=patterns.get("electrical"))
    sample_ds.extract_temp_props(patterns=patterns.get("temperature"))

    assert sample_ds.extracted_temp == temp_result_dict
    assert sample_ds.extracted_elec == elec_result_dict

def test_two():

    with open("/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/patterns.yaml", "r") as stream:
        try:
            patterns = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    elec_result_dict = {'eff': {'cols': [0], 'vals': ['21.40%', '21.20%', '21.00%', '20.80%', '20.60%', '20.40%', '20.20%']}, 'isc': {'cols': [5], 'vals': ['9.675A', '9.646A', '9.626A', '9.608A', '9.587A', '9.572A', '9.570A']}, 'voc': {'cols': [4], 'vals': ['0.671V', '0.671V', '0.666V', '0.663V', '0.659V', '0.654V', '0.648V']}, 'impp': {'cols': [3], 'vals': ['9.156A', '9.104A', '9.084A', '9.059A', '9.041A', '9.031A', '9.027A']}, 'vmpp': {'cols': [2], 'vals': ['0.571V', '0.569V', '0.565V', '0.561V', '0.557V', '0.552V', '0.547V']}, 'pmpp': {'cols': [1], 'vals': ['5.23W', '5.18W', '5.13W', '5.08W', '5.03W', '4.98W', '4.94W']}, 'ff': {'cols': None, 'vals': None}}
    temp_result_dict = {'isc': {'rows': [1], 'vals': ['+0.043%/℃']}, 'voc': {'rows': [2], 'vals': ['-0.30%/℃']}, 'pmpp': {'rows': [3], 'vals': ['-0.38%/℃']}}


    sample_ds = Datasheet(
        path_to_excel="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/tests/files/two.xlsx",
        path_to_clf="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/nb_classifier.pickle",
        path_to_vec="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/vectoriser.pickle"
    )
    
    sample_ds.extract_electrical_props(patterns=patterns.get("electrical"))
    sample_ds.extract_temp_props(patterns=patterns.get("temperature"))

    assert sample_ds.extracted_temp == temp_result_dict
    assert sample_ds.extracted_elec == elec_result_dict

def test_three():

    with open("/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/patterns.yaml", "r") as stream:
        try:
            patterns = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    elec_result_dict = {'eff': {'cols': [1], 'vals': ['22.80%', '22.70%', '22.60%', '22.50%', '22.40%', '22.30%', '22.20%', '22.10%', '22.00%', '21.90%', '21.80%']}, 'isc': {'cols': [6], 'vals': [11.124, '11.119', '11.103', '11.089', '11.077', '11.060', '11.046', '11.033', '11.025', '11.008', '10.989']}, 'voc': {'cols': [5], 'vals': ['0.6803', '0.6784', '0.677', '0.6761', '0.6746', '0.6732', '0.6699', '0.6683', '0.6662', '0.6641', '0.6619']}, 'impp': {'cols': [4], 'vals': ['10.550', '10.544', '10.529', '10.515', '10.502', '10.488', '10.485', '10.472', '10.464', '10.451', '10.437']}, 'vmpp': {'cols': [3], 'vals': ['0.5925', '0.5902', '0.5885', '0.5866', '0.5848', '0.5829', '0.5805', '0.5786', '0.5764', '0.5745', '0.5726']}, 'pmpp': {'cols': [2], 'vals': ['6.25', '6.22', '6.20', '6.17', '6.14', '6.11', '6.09', '6.06', '6.03', '6.00', '5.98']}, 'ff': {'cols': None, 'vals': None}}
    temp_result_dict = {'isc': {'rows': [2], 'vals': ['0.07']}, 'voc': {'rows': [1], 'vals': ['-0.36']}, 'pmpp': {'rows': [3], 'vals': []}}

    sample_ds = Datasheet(
        path_to_excel="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/tests/files/three.xlsx",
        path_to_clf="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/nb_classifier.pickle",
        path_to_vec="/Users/moeezmalik/Documents/Main/work/fraunhofer/thesis/Lightning-Table/validation/vectoriser.pickle"
    )
    
    sample_ds.extract_electrical_props(patterns=patterns.get("electrical"))
    sample_ds.extract_temp_props(patterns=patterns.get("temperature"))

    assert sample_ds.extracted_temp == temp_result_dict
    assert sample_ds.extracted_elec == elec_result_dict