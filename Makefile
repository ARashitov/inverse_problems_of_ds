# Project utilities
env_create:
	conda create -n inverse_problems_of_ds python=3.10 -y

env_configure: env_install_dependencies env_install_jupyter_extensions env_install_precommit_hooks
	echo "Environment is configured"

env_install_precommit_hooks:
	pre-commit install && pre-commit install --hook-type commit-msg

env_install_dependencies:
	pip3 install --upgrade pip \
	&& pip3 install wheel poetry \
	&& poetry install

env_install_jupyter_extensions:
	jupyter contrib nbextension install --sys-prefix \
	&& jupyter nbextension install --user https://rawgithub.com/minrk/ipython_extensions/master/nbextensions/toc.js \
	&& jupyter nbextension enable --py widgetsnbextension \
	&& jupyter nbextension enable codefolding/main \
	&& jupyter nbextension enable --py keplergl \
	&& jupyter nbextension enable spellchecker/main \
	&& jupyter nbextension enable toggle_all_line_numbers/main \
	&& jupyter nbextension enable hinterland/hinterland \
	&& pip install jupyterthemes && jt -t oceans16

env_delete:
	conda remove --name inverse_problems_of_ds --all -y

run_uvicorn_local:
	uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload --log-config ./local_log_config.ini

run_uvicorn_remote:
	uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload --log-config ./remote_log_config.ini

run_test:
	kedro test

run_update_kedro_context:
	python3 conf/context_management/main.py

run_jupyter:
	jupyter-notebook --ip 0.0.0.0 --no-browser

run_precommit:
	pre-commit run --all-files
