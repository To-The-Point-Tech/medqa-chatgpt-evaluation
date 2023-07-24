import os
import requests


def ttp_renew_token():
    response = requests.post(
        'https://tothepoint.tech/api/token', 
        headers={
            'accept': 'application/json',
            "Access-Control-Allow-Credentials": "*",
            "Content-Type": "application/x-www-form-urlencoded",
        }, 
        data={
            "username": os.environ["TOTHEPOINT_USERNAME"],
            "password": os.environ["TOTHEPOINT_PASSWORD"],
        },
        verify=False
    )
    token = response.json()["access_token"]
    return token


def ttp_validate_token(token: str):
    response = requests.post(
        'https://tothepoint.tech/api/validate', 
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "access_token": token,
            "token_type": "bearer",
        },
        verify=False
    ).json()
    if "detail" in response and "Could not validate credentials" in response["detail"]:
        return False
    return True


def ttp_retrieve(query: str, token: str, k: int = 5):
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    response = requests.post(
        'https://tothepoint.tech/api/retrieve', 
        headers=headers, 
        json={
            "query": query,
            "nearest_neighbours": k
        },
        verify=False
    )
    return response.json()