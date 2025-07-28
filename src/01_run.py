# src/01_run_refactored.py
from analysis.data_loader import get_raw_data, preprocess_data
from analysis.modeling import build_design_matrices, fit_lmm_reml
from analysis.reporting import calculate_model_statistics, print_summary
from analysis.visualization import (
    visualize_raw_data,
    visualize_design_matrices,
    visualize_convergence,
    visualize_results
)

def main():
	"""LMM 분석 파이프라인을 실행합니다."""
	# 1. 데이터 로딩, 전처리 및 시각화
	raw_data = get_raw_data()
	proc_data = preprocess_data(raw_data)
	visualize_raw_data(proc_data)

	# 2. 디자인 행렬 생성 및 시각화
	X, Z = build_design_matrices(
		proc_data['time'],
		proc_data['student_indices'],
		proc_data['n_obs'],
		proc_data['n_students']
	)
	visualize_design_matrices(X, Z)

	# 3. 모델 피팅 및 최적화 과정 시각화
	optimization_result, history = fit_lmm_reml(proc_data['y'], X, Z)
	visualize_convergence(history)

	# 4. 최종 통계량 계산 및 결과 출력/시각화
	if optimization_result.success:
		final_results = calculate_model_statistics(optimization_result, proc_data['y'], X, Z)
		print_summary(final_results)
		visualize_results(final_results, proc_data)
	else:
		print("모델 최적화에 실패했습니다.")
		print(optimization_result.message)


if __name__ == '__main__':
	main()