import logging
from pathlib import Path
import configs
import templates_handler


logging.basicConfig(format='[ %(asctime)s ][ %(levelname)s ]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


for _env in configs.ENVIRONMENTS:

    # 1. Credentials export
    cred_fpath = "conf/context_management/templates/cred_template.yaml"
    credentials_template = templates_handler.read(cred_fpath)
    credentials = templates_handler.interpolate(credentials_template, configs.CREDENTIALS_MAPPINGS)
    credentials_location = f"conf/{_env}/credentials.yaml"
    # NOTE: just make sure, that target directory exists
    Path(credentials_location).parent.mkdir(parents=True, exist_ok=True)
    templates_handler.export(credentials, credentials_location)

    # 2. Catalog export
    catalog_template = templates_handler.read("conf/context_management/templates/catalog.yaml")
    configs.CATALOG_MAPPINGS['{env}'] = _env
    catalog = templates_handler.interpolate(catalog_template, configs.CATALOG_MAPPINGS)
    catalog_location = f"conf/{_env}/catalog.yaml"
    templates_handler.export(catalog, catalog_location)

    # 3. Parameters export
    params_template = templates_handler.read("conf/context_management/templates/parameters.yaml")
    parameters_location = f"conf/{_env}/parameters.yaml"
    templates_handler.export(params_template, parameters_location)

    logging.info(f"Finish export context for {_env} environment\n")

templates_handler.export(credentials, "conf/base/credentials.yaml")
templates_handler.export(credentials, "conf/local/credentials.yaml")
