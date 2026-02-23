from composio import ComposioToolSet
toolset = ComposioToolSet(api_key="35adjvgxl35dn3hz5o42h")

integration = toolset.get_integration(id="0418a8ed-6017-4fdc-99de-d4861b784529")
# Collect auth params from your users
print(integration.expectedInputFields)

connection_request = toolset.initiate_connection(
    integration_id=integration.id,
    entity_id="default",
)

# Redirect step require for OAuth Flow
print(connection_request.redirectUrl)
print(connection_request.connectedAccountId)


#(venv) abhay@Abhays-MacBook-Air composio_test % python3 agent/integration.py
# [2025-07-07 16:30:21,212][INFO] Actions cache is outdated, refreshing cache...
# []
# https://backend.composio.dev/api/v3/s/TodnaFaa
# 8a2f7843-ca6b-41c0-bb98-21758e63104b