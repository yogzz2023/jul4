import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom
        self.first_step_flag = False
        self.second_step_flag = True

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        # print("Initialized filter state:")
        # print("Sf:", self.Sf)
        # print("Pf:", self.Pf)
        
    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        # print("Predicted filter state:")
        # print("Sp:", self.Sp)
        # print("Pp:", self.Pp)

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        # print("Updated filter state:")
        # print("Sf:", self.Sf)
        # print("Pf:", self.Pf)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan(y / x)    

    if x > 0.0:                
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az       
        
    az = az * 180 / 3.14 

    if az < 0.0:
        az = 360 + az
    
    if az > 360:
        az = az - 360   
      
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       
        
        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]
    
        if az[i] > 360:
            az[i] = az[i] - 360

    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def form_measurement_groups(measurements, max_time_diff=50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]
    
    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]
    
    if current_group:
        measurement_groups.append(current_group)
        
    return measurement_groups

def chi_square_clustering(group, filter_instance):
    clusters = []
    for i, measurement in enumerate(group):
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if filter_instance.gating(Z).item():
            clusters.append(measurement)
    return clusters

# Integration of clustering and association functions

state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

print("Covariance Matrix:\n", cov_matrix)
print("Chi-squared Threshold:", chi2_threshold)

def generate_hypotheses(tracks, reports):
    num_tracks = len(tracks)
    num_reports = len(reports)
    base = num_reports + 1
    
    hypotheses = []
    for count in range(base**num_tracks):
        hypothesis = []
        for track_idx in range(num_tracks):
            report_idx = (count // (base**track_idx)) % base
            hypothesis.append((track_idx, report_idx - 1))
        
        if is_valid_hypothesis(hypothesis):
            hypotheses.append(hypothesis)
    
    return hypotheses

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx][:3], reports[report_idx][:3], cov_inv)
                prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities


def get_association_weights(hypotheses, probabilities):
    num_tracks = len(hypotheses[0])
    association_weights = [[] for _ in range(num_tracks)]
    
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                association_weights[track_idx].append((report_idx, prob))
    
    for track_weights in association_weights:
        track_weights.sort(key=lambda x: x[0])
        report_probs = {}
        for report_idx, prob in track_weights:
            if report_idx not in report_probs:
                report_probs[report_idx] = prob
            else:
                report_probs[report_idx] += prob
        track_weights[:] = [(report_idx, prob) for report_idx, prob in report_probs.items()]
    
    return association_weights


def find_max_associations(hypotheses, probabilities, reports):
    num_reports = len(reports)
    association_probs = np.zeros(num_reports)
    max_associations = -1 * np.ones(num_reports, dtype=int)
    
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                if prob > association_probs[report_idx]:
                    association_probs[report_idx] = prob
                    max_associations[report_idx] = track_idx
    
    return max_associations, association_probs

# Main code execution
file_path = 'ttk_50.csv'
measurements = read_measurements_from_csv(file_path)
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_50.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values
measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)


filtered_range = []
filtered_azimuth = []
filtered_elevation = []
filtered_time = []

kalman_filter = CVFilter()
targets = []

# Initialize a counter for the number of measurements processed
measurement_count = 0
previous_measurement = None

# Main Kalman Filter execution flow
for group_index, group in enumerate(measurement_groups):
    for i, (x, y, z, mt) in enumerate(group):
        r, az, el = cart2sph(x, y, z)
        Z = np.array([[x], [y], [z]])

        if measurement_count == 0:  # First measurement
            kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            targets.append(kalman_filter.Sp)
            kalman_filter.first_step_flag = True
            kalman_filter.second_step_flag = False
            previous_measurement = (x, y, z, mt)
        elif measurement_count == 1:  # Second measurement, regardless of group
            prev_x, prev_y, prev_z, prev_mt = previous_measurement
            dt = mt - prev_mt
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            vz = (z - prev_z) / dt
            print("vz:",vz)
            kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
            targets.append(kalman_filter.Sp)
            kalman_filter.second_step_flag = False
        elif measurement_count >= 2:  # Starting from the third measurement
            kalman_filter.predict_step(mt)
            clusters = chi_square_clustering(group, kalman_filter)
            if len(clusters) > 0:
                cluster_tracks = np.array([target[:3].flatten() for target in targets])
                print("ccccccccccttttttt",cluster_tracks)
                cluster_reports = np.array(clusters)
                print("ccccccccccrrr",cluster_reports)
                hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
                print("dddwedesahyh",hypotheses)
                probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
                association_weights = get_association_weights(hypotheses, probabilities)
                max_associations, max_probs = find_max_associations(hypotheses, probabilities, cluster_reports)
                for report_idx, association in enumerate(max_associations):
                    if association != -1:
                        best_measurement = clusters[report_idx]
                        Z = np.array([[best_measurement[0]], [best_measurement[1]], [best_measurement[2]]])
                        kalman_filter.update_step(Z)

        filtered_range.append(r)
        filtered_azimuth.append(az)
        filtered_elevation.append(el)
        filtered_time.append(mt)

        measurement_count += 1  # Increment the measurement count

# Save the filtered data to a new CSV file
filtered_data = pd.DataFrame({
    'filtered_range': filtered_range,
    'filtered_azimuth': filtered_azimuth,
    'filtered_elevation': filtered_elevation,
    'filtered_time': filtered_time
})
filtered_data.to_csv('filtered_data.csv', index=False)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
plt.scatter(filtered_time, filtered_range, label='filtered range (code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[0], label='filtered range (track id 31)', color='red', marker='*')
plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='measured range (track id 31)', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
plt.scatter(filtered_time, filtered_azimuth, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
plt.scatter(filtered_time, filtered_range, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()
