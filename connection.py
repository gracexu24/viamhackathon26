import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()


async def connect():
    api_key = os.environ["VIAM_API_KEY"]
    api_key_id = os.environ["VIAM_API_KEY_ID"]
    address = os.environ["VIAM_ADDRESS"]

    options = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id,
    )

    return await RobotClient.at_address(address, options)