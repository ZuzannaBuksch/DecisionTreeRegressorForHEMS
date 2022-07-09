import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_data_from_json(file_name):
    data = open(file_name, 'r')
    return json.load(data)

def save_data_to_json(file_name, data):
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

def get_dataset(all_raports):
    raports_with_classes = []

    for number_of_raport in range(0, len(all_raports)):
        single_raport = all_raports[number_of_raport]
        grid_price = single_raport["public_grid_price"]
        exchange_price = list(single_raport["exchange_data"].values())[0] 
        sec_window_energy_from_public_grid = list(single_raport["public_grid_data"].values())[1]

        if exchange_price < grid_price:
            raport_with_class = create_report_with_class(single_raport, sec_window_energy_from_public_grid)
            raports_with_classes.append(raport_with_class)
        else:
            raport_with_class = create_report_with_class(single_raport, 0.0)
            raports_with_classes.append(raport_with_class)
    return raports_with_classes

def create_report_with_class(single_raport, obj_class):
    #-----------------------------------fields from report-------------------------------
    energy_storage_after_first_window = list(single_raport['energy_storage'].values())[0]
    total_storage_capacity = single_raport['total_storage_capacity']
    energy_generation_first_window = list(single_raport['energy_generation'].values())[0]
    energy_usage_first_window = list(single_raport['energy_usage'].values())[0]
    surplus_after_first_window = list(single_raport['surplus_data'].values())[0]
    used_public_grid_in_first_window = list(single_raport['public_grid_data'].values())[0]
    exchange_price_at_the_beginning = list(single_raport['exchange_data'].values())[0]
    initial_surplus_value = single_raport['initial_grid_surplus_value']
    initial_storage_value = single_raport['initial_storage_charge_value']
    generation_power = single_raport['generation_power']
    weather_data_sec_window = list(single_raport['weather_data'].values())[12:]
    sec_window_generation = calculate_time_window_photovoltaics_generation(weather_data_sec_window, generation_power)
    #-------------------------------------------------------------------------------------

    battery_charge = energy_storage_after_first_window / (total_storage_capacity + 0.00001)
    battery_charge = battery_charge[0]
    generation_to_usage_ratio = energy_generation_first_window / (energy_usage_first_window + 0.00001)
    initial_suprlus_and_storage_to_usage_ratio = (initial_surplus_value + initial_storage_value) / (energy_usage_first_window + 0.00001)
    if_taken_from_public_grid = 1 if used_public_grid_in_first_window != 0 else 0
    if_taken_from_storage =  0 if initial_storage_value > energy_storage_after_first_window else 1

    report = {
                'battery_charge' :  battery_charge,
                'generation_to_usage_ratio' : generation_to_usage_ratio,
                'initial_suprlus_and_storage_to_usage_ratio' : initial_suprlus_and_storage_to_usage_ratio,
                'if_taken_from_public_grid' : if_taken_from_public_grid,
                'exchange_price' : exchange_price_at_the_beginning,
                'surplus_after_first_window': surplus_after_first_window,
                'if_taken_from_storage' : if_taken_from_storage,
                'sec_window_generation' : sec_window_generation,
                'class': obj_class}
    return report

def calculate_time_window_photovoltaics_generation(weather_data, generation_power):
    sum_of_energy_in_kwh = 0.0
    for value in weather_data:
        solar_radiation = value['forecast']
        diff_in_hours = 0.0833
        solar_radiation_coefficient = solar_radiation / 1000
        output_power = generation_power * solar_radiation_coefficient * (1 - 0.05)
        output_power_in_kwh = output_power / 1000 * diff_in_hours
        sum_of_energy_in_kwh += output_power_in_kwh
    return sum_of_energy_in_kwh

def show_error_rate(y_test, y_pred):
    x = []
    error = []
    for i in range (0,len(y_test)):
        x.append(i)
        diff = float(y_test[i]) - float(y_pred[i])
        error.append(diff)
    
    plt.plot(x, error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Difference of actual and predicted amount of energy that we should buy on the exchange for the next hour [kWh]')
    plt.xlabel('Single time window')
    plt.ylabel('Difference of energy [kWh]')
    plt.show()

def show_accuracy_figure(y_test, y_pred):
    x = []
    for i in range (0,len(y_test)):
        x.append(i)
    plt.plot(x, [float(x) for x in y_test], label = 'Actual')
    plt.plot(x, [float(x) for x in y_pred], label = 'Predicted')
    plt.title('Amount of energy that we should buy on the exchange for the next hour [kWh]')
    plt.xlabel('Single time window')
    plt.ylabel('Energy [kWh]')
    plt.legend()
    plt.show()

def show_correlations_map(df):
    # https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
    correlations = df.corr()
    # annot=True displays the correlation values
    sns.heatmap(correlations, annot=True).set(title='Correlations of variables')
    plt.show()

def save_model_to_file(regressor):
    filename = 'regressor_model.sav'
    pickle.dump(regressor, open(filename, 'wb'))

def calculate_mean_absolute_error(y_test, y_pred):
    mean_error = mean_absolute_error(y_test, y_pred)
    print('Mean_absolute_error: ', round(mean_error, 4))

def main():
    all_data = get_data_from_json('random_data.json')
    data = get_dataset(all_data)
    dataset =  pd.DataFrame.from_dict(data, orient='columns')
    
    #show_correlations_map(dataset)
    
    numer_of_columns = dataset.shape[1]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, numer_of_columns -1].values   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.03, random_state = 2) 
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    #show_error_rate(y_test, y_pred)
    #show_accuracy_figure(y_test, y_pred)
    save_model_to_file(regressor)
    calculate_mean_absolute_error(y_test, y_pred)

    loaded_model = pickle.load(open('regressor_model.sav', 'rb'))
    y_pred = loaded_model.predict(X_test)
    calculate_mean_absolute_error(y_test, y_pred)
    
if __name__ == "__main__":
    main()
