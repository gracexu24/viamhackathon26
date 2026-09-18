import asyncio

from connection import connect
from viam.components.arm import Arm
from viam.proto.common import Pose


async def main():
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")

        current = await arm.get_end_position()
        print(f"Current end position: {current}")

        target = Pose(
            x = 188.62215715380211,
            y = -253.68014729608331,
            z = 492.62546955868106,
            o_x = 0.12223813513169798,
            o_y = -0.76280727003391569,
            o_z = -0.63496685512153228,
            theta = 177.75629710721796
        )

        await arm.move_to_position(pose=target)
        print(f"Moved to: {target}")


if __name__ == "__main__":
    asyncio.run(main())




















