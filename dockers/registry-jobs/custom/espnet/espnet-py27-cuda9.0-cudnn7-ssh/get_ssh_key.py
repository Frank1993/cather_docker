#!/usr/bin/env python
import json
import pprint
import sys
import re
from azure.keyvault import KeyVaultClient
from azure.common.credentials import ServicePrincipalCredentials

vault_name = 'gcrauthvault'
client_id = '2f778170-3d10-43a6-979a-da1390d5964c'
secret = 'IGQZeHePZ8/6Z+CF8ZQ3Z+qdDQjV/6CTJ7tOEMvFBT0='
tenant = '72f988bf-86f1-41af-91ab-2d7cd011db47'
# We need at least one arg for the username
if len(sys.argv) == 1:
  print ('not a valid username')
  sys.exit()

# Get our username, and lowercase it
username = sys.argv[1].lower()

# Check if this all letters
if not re.search('[a-z]',username):
  print ('not a valid username')
  sys.exit()
if re.search('\.',username):
  alias = username.split('.')[1]
else:
  alias = username

# Using gcrauthvaultro
KEY_VAULT_URI="https://{0}.vault.azure.net/".format(vault_name)
credentials = ServicePrincipalCredentials(
    client_id=client_id,
    secret=secret,
    tenant=tenant
)

client = KeyVaultClient(
    credentials
)
try:
  secret = client.get_secret(KEY_VAULT_URI,alias,'')
except:
  pass

try:
  secret.value
except NameError:
  print ('No value returned from KeyVault')
  sys.exit(1)
else:
  print (secret.value)