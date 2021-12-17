run_server:
		python -m app

run_client:
		python -m streamlit run app/frontend.py --server.port 8001

run_app: run_server run_client