import asyncio
from bleak import BleakScanner

async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Address: {device.address}, Name: {device.name}, UUIDs: {device.metadata.get('uuids', 'No UUIDs available')}")
        
asyncio.run(main())
