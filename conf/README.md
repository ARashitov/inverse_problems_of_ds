# What is this for?

This folder should be used to store configuration files used by Kedro or by separate tools.

This file can be used to provide users with instructions for how to reproduce local configuration with their own credentials. You can edit the file however you like, but you may wish to retain the information below and add your own section in the [Instructions](#Instructions) section.

## context_managmenet

The `context_managmenet` folder should be used for configuration of kedro context which are: `catalog.yaml`, `parameters.yaml` & `credentials.yaml`

> *Note:* Please do not check in any `credentials.yaml` configuration to version control.

## Instructions


if you want to extend context navigate to `conf/context_management/templates` directory and do extension


## Find out more
You can find out more about configuration from the [user guide documentation](https://kedro.readthedocs.io/en/stable/04_user_guide/03_configuration.html).
