"""
Nexum v2.0 Communication Protocol
Handles inter-module communication and data exchange
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import websockets
import pyzmq
from pyzmq import Context


class MessageType(Enum):
    COMMAND = "command"
    DATA = "data"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    id: str
    type: MessageType
    source: str
    destination: str
    payload: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            source=data["source"],
            destination=data["destination"],
            payload=data["payload"],
            timestamp=data["timestamp"]
        )


class CommunicationProtocol:
    """
    Central communication protocol for Nexum v2.0
    Handles message routing between modules
    """
    
    def __init__(self, node_id: str = "kernel"):
        self.node_id = node_id
        self.context = Context()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = asyncio.Queue()
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # ZMQ sockets
        self.publisher = self.context.socket(pyzmq.PUB)
        self.subscriber = self.context.socket(pyzmq.SUB)
        
        # WebSocket server
        self.websocket_server = None
        self.websocket_clients = set()
        
    async def initialize(self, zmq_port: int = 5555, ws_port: int = 8765) -> bool:
        """Initialize communication channels"""
        try:
            # Initialize ZMQ
            self.publisher.bind(f"tcp://*:{zmq_port}")
            self.subscriber.connect(f"tcp://localhost:{zmq_port}")
            self.subscriber.setsockopt_string(pyzmq.SUBSCRIBE, "")
            
            # Initialize WebSocket server
            self.websocket_server = await websockets.serve(
                self._handle_websocket_client,
                "localhost",
                ws_port
            )
            
            self.running = True
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            asyncio.create_task(self._heartbeat_loop())
            
            self.logger.info(f"Communication protocol initialized on ports {zmq_port} (ZMQ) and {ws_port} (WS)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication protocol: {e}")
            return False
    
    async def _handle_websocket_client(self, websocket, path):
        """Handle incoming WebSocket connections"""
        self.websocket_clients.add(websocket)
        client_id = str(uuid.uuid4())
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg = Message.from_dict(data)
                    await self.message_queue.put(msg)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid message format")
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
    
    async def _process_messages(self):
        """Process messages from queue"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._route_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _route_message(self, message: Message):
        """Route message to appropriate handlers"""
        # Handle broadcasts
        if message.destination == "broadcast":
            await self._broadcast_message(message)
        
        # Handle specific destinations
        elif message.destination in self.subscribers:
            for callback in self.subscribers[message.destination]:
                try:
                    await callback(message)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")
        
        # Handle kernel messages
        elif message.destination == "kernel":
            await self._handle_kernel_message(message)
    
    async def _broadcast_message(self, message: Message):
        """Broadcast message to all subscribers"""
        # ZMQ broadcast
        self.publisher.send_json(message.to_dict())
        
        # WebSocket broadcast
        if self.websocket_clients:
            message_str = json.dumps(message.to_dict())
            await asyncio.gather(
                *[client.send(message_str) for client in self.websocket_clients],
                return_exceptions=True
            )
    
    async def _handle_kernel_message(self, message: Message):
        """Handle messages destined for the kernel"""
        if message.type == MessageType.COMMAND:
            self.logger.info(f"Received command: {message.payload}")
        elif message.type == MessageType.STATUS:
            self.logger.info(f"Status update: {message.payload}")
        elif message.type == MessageType.ERROR:
            self.logger.error(f"Error from {message.source}: {message.payload}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.running:
            heartbeat = Message(
                id=str(uuid.uuid4()),
                type=MessageType.HEARTBEAT,
                source=self.node_id,
                destination="broadcast",
                payload={"status": "alive", "timestamp": time.time()},
                timestamp=time.time()
            )
            await self._broadcast_message(heartbeat)
            await asyncio.sleep(10)  # Heartbeat every 10 seconds
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message to WebSocket client"""
        error_msg = Message(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            source=self.node_id,
            destination="websocket_client",
            payload={"error": error_message},
            timestamp=time.time()
        )
        await websocket.send(json.dumps(error_msg.to_dict()))
    
    def subscribe(self, destination: str, callback: Callable):
        """Subscribe to messages for a specific destination"""
        if destination not in self.subscribers:
            self.subscribers[destination] = []
        self.subscribers[destination].append(callback)
    
    async def send_message(self, destination: str, message_type: MessageType, payload: Dict[str, Any]) -> str:
        """Send a message to a specific destination"""
        message = Message(
            id=str(uuid.uuid4()),
            type=message_type,
            source=self.node_id,
            destination=destination,
            payload=payload,
            timestamp=time.time()
        )
        
        await self.message_queue.put(message)
        return message.id
    
    async def send_command(self, destination: str, command: str, parameters: Dict[str, Any] = None) -> str:
        """Send a command message"""
        payload = {"command": command}
        if parameters:
            payload.update(parameters)
        
        return await self.send_message(destination, MessageType.COMMAND, payload)
    
    async def send_data(self, destination: str, data_type: str, data: Any) -> str:
        """Send a data message"""
        payload = {"data_type": data_type, "data": data}
        return await self.send_message(destination, MessageType.DATA, payload)
    
    async def send_status(self, destination: str, status: Dict[str, Any]) -> str:
        """Send a status message"""
        return await self.send_message(destination, MessageType.STATUS, status)
    
    def shutdown(self):
        """Shutdown the communication protocol"""
        self.running = False
        self.publisher.close()
        self.subscriber.close()
        self.context.term()


class ModuleCommunicator:
    """
    Helper class for modules to communicate with the kernel
    """
    
    def __init__(self, module_name: str, kernel_host: str = "localhost", kernel_port: int = 8765):
        self.module_name = module_name
        self.kernel_host = kernel_host
        self.kernel_port = kernel_port
        self.websocket = None
        self.message_handlers: Dict[str, Callable] = {}
    
    async def connect(self) -> bool:
        """Connect to the kernel"""
        try:
            uri = f"ws://{self.kernel_host}:{self.kernel_port}"
            self.websocket = await websockets.connect(uri)
            
            # Start message handler
            asyncio.create_task(self._message_loop())
            
            # Send initialization message
            await self.send_status({"status": "connected", "module": self.module_name})
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to kernel: {e}")
            return False
    
    async def _message_loop(self):
        """Handle incoming messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg = Message.from_dict(data)
                
                # Route to appropriate handler
                if msg.type.value in self.message_handlers:
                    await self.message_handlers[msg.type.value](msg)
                    
        except websockets.exceptions.ConnectionClosed:
            logging.warning("Connection to kernel closed")
    
    def on_message(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    async def send_message(self, destination: str, message_type: MessageType, payload: Dict[str, Any]):
        """Send a message"""
        if not self.websocket:
            raise ConnectionError("Not connected to kernel")
        
        message = Message(
            id=str(uuid.uuid4()),
            type=message_type,
            source=self.module_name,
            destination=destination,
            payload=payload,
            timestamp=time.time()
        )
        
        await self.websocket.send(json.dumps(message.to_dict()))
    
    async def send_command(self, destination: str, command: str, parameters: Dict[str, Any] = None):
        """Send a command"""
        payload = {"command": command}
        if parameters:
            payload.update(parameters)
        
        await self.send_message(destination, MessageType.COMMAND, payload)
    
    async def send_data(self, destination: str, data_type: str, data: Any):
        """Send data"""
        payload = {"data_type": data_type, "data": data}
        await self.send_message(destination, MessageType.DATA, payload)
    
    async def send_status(self, status: Dict[str, Any]):
        """Send status update"""
        await self.send_message("kernel", MessageType.STATUS, status)
