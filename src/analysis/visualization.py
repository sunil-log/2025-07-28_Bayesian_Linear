# src/analysis/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_raw_data(proc_data: dict):
	"""1. 데이터 탐색: 학생별 점수 변화 추이 시각화"""
	student_names_array = np.array(proc_data['student_names'])
	df = pd.DataFrame({
		'student_id': student_names_array[proc_data['student_indices']],
		'time': proc_data['time'],
		'score': proc_data['y']
	})

	plt.figure(figsize=(10, 6))
	sns.lineplot(data=df, x='time', y='score', hue='student_id', marker='o')
	plt.title('1. Raw Data: Student Score Trajectories')
	plt.xlabel('Time (months)')
	plt.ylabel('Score')
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend(title='Student ID')
	plt.savefig('1_raw_data_trajectories.png')
	plt.close()
	print("[INFO] Saved '1_raw_data_trajectories.png'")

def visualize_design_matrices(X: np.ndarray, Z: np.ndarray):
	"""2. 디자인 행렬 구조 시각화"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
	sns.heatmap(X, ax=ax1, cmap='viridis', annot=True, fmt='.0f', cbar=False)
	ax1.set_title('2a. Fixed Effects Design Matrix (X)')
	ax1.set_xlabel('Fixed Effects (Intercept, Time)')
	ax1.set_ylabel('Observations')
	sns.heatmap(Z, ax=ax2, cmap='YlGnBu', cbar=True)
	ax2.set_title('2b. Random Effects Design Matrix (Z)')
	ax2.set_xlabel('Random Effects (Intercepts & Slopes per Student)')
	ax2.set_ylabel('Observations')
	plt.tight_layout()
	plt.savefig('2_design_matrices_heatmap.png')
	plt.close()
	print("[INFO] Saved '2_design_matrices_heatmap.png'")

def visualize_convergence(convergence_history: list):
	"""3. 최적화 과정 시각화"""
	plt.figure(figsize=(10, 6))
	plt.plot(convergence_history, marker='.', linestyle='-')
	plt.title('3. Optimization Convergence Plot')
	plt.xlabel('Iteration')
	plt.ylabel('Negative REML Log-Likelihood')
	plt.grid(True)
	plt.savefig('3_reml_convergence.png')
	plt.close()
	print("[INFO] Saved '3_reml_convergence.png'")

def visualize_results(final_results: dict, proc_data: dict):
	"""4. 최종 결과 시각화: 예측선 및 분산 성분"""
	student_names_array = np.array(proc_data['student_names'])
	df = pd.DataFrame({
		'student_id': student_names_array[proc_data['student_indices']],
		'time': proc_data['time'],
		'score': proc_data['y']
	})
	beta_intercept, beta_time = final_results['beta_est']
	blups = final_results['blups']

	plt.figure(figsize=(12, 8))
	sns.scatterplot(data=df, x='time', y='score', hue='student_id', s=100, alpha=0.7)
	time_range = np.linspace(df['time'].min(), df['time'].max(), 100)
	plt.plot(time_range, beta_intercept + beta_time * time_range, 'k--', linewidth=3,
	         label='Population Fit (Fixed Effects)')

	for i, student_name in enumerate(proc_data['student_names']):
		student_blup_intercept, student_blup_slope = blups[i]
		individual_intercept = beta_intercept + student_blup_intercept
		individual_slope = beta_time + student_blup_slope
		color = sns.color_palette()[i % len(sns.color_palette())]
		plt.plot(time_range, individual_intercept + individual_slope * time_range, color=color, linewidth=1.5)

	plt.title('4a. Fitted Lines Plot: Population and Individual Predictions')
	plt.xlabel('Time (months)')
	plt.ylabel('Score')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.savefig('4a_fitted_lines_plot.png')
	plt.close()
	print("[INFO] Saved '4a_fitted_lines_plot.png'")

	variances = final_results['variances']
	var_df = pd.DataFrame.from_dict(variances, orient='index', columns=['Variance']).reset_index()
	var_df = var_df.rename(columns={'index': 'Component'})
	plt.figure(figsize=(8, 6))
	sns.barplot(data=var_df, x='Component', y='Variance')
	plt.title('4b. Estimated Variance Components')
	plt.ylabel('Variance')
	plt.xlabel('Random Effect Component')
	plt.tight_layout()
	plt.savefig('4b_variance_components.png')
	plt.close()
	print("[INFO] Saved '4b_variance_components.png'")