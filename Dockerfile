# 여기서는 2.1.0-cuda12.1-cudnn8-runtime을 예로 들었습니다.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 필요한 패키지를 설치합니다.
RUN pip install torchvision \
				pandas \
				matplotlib \
				seaborn \
				Pillow \
				plotly \
				scikit-learn \
				statsmodels \
				openai \
				umap-learn \
				shap \
				tables \
				tabulate \
				PyMySQL \
				pyyaml \
				torchmetrics \
				optuna \
				rich \
				pytorch-ignite \
				tensordict \
				pysindy \
				optht \
				xgboost


# 컨테이너 실행시 자동으로 실행될 명령어
CMD ["bash"]

# docker build -t sac/lightning .
