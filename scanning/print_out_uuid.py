'''記得先用藍芽連上'''
import asyncio
from bleak import BleakClient

async def discover_services_and_characteristics(mac_address):
    async with BleakClient(mac_address) as client:
        # 检查是否成功连接
        print(f"Connected: {client.is_connected}")

        # 获取并打印所有服务和特性
        services = await client.get_services()
        for service in services:
            print(f"Service: {service.uuid} ({service.description})")
            for characteristic in service.characteristics:
                print(f"  Characteristic: {characteristic.uuid} ({characteristic.description})")

if __name__ == "__main__":
    #Xsens DOT3
    # mac_address = "D4:22:CD:00:38:55"
    #Xsens DOT1 
    # mac_address = "D4:22:CD:00:38:5A"  
    #Xsens DOT2
    mac_address = "D4:22:CD:00:38:5B"  
    asyncio.run(discover_services_and_characteristics(mac_address))

