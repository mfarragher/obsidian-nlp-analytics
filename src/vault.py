import os
import obsidiantools.api as otools
from dotenv import load_dotenv


load_dotenv()


def get_vault(env_variable):
    kwargs = _get_vault_args(env_variable)
    vault = otools.Vault(
        os.environ.get(env_variable),
        **kwargs).connect().gather()
    return vault


def _get_vault_args(env_variable):
    if env_variable == 'MPHIL_VAULT_DIR':
        return {'include_subdirs': ['PEpi', 'SHDS', 'SHDSpt2',
                                    'ADA', 'PPH', 'RS',
                                    'ABHDS', 'GEpi', 'IML', 'AML', 'HDSBS', 'AG'],
                'include_root': False}
    if env_variable == 'FILM_NOIR_VAULT_DIR':
        return {'include_root': True}
    raise ValueError('Specify env variable for vault location')
