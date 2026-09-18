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
            x=-58,
            y=-172 ,
            z=690,
            o_x= -0.53400040109057245,
            o_y= -0.84506158421718891,
            o_z= -0.026729955395446758,
            theta= -177.76549443708427,
        )

        await arm.move_to_position(pose=target)
        print(f"Moved to: {target}")


if __name__ == "__main__":
    asyncio.run(main())