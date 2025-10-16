#!/usr/bin/env python3
"""
Debug test for collision logic and agent state management.
This will help us understand why agents get stuck in front of storage units.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.warehouse_env import WarehouseEnv
from environment.employee import Employee, EmployeeState
from environment.warehouse_grid import WarehouseGrid, CellType
import numpy as np

def create_simple_test_env():
    """Create a minimal test environment"""
    env = WarehouseEnv(
        grid_width=15, 
        grid_height=15, 
        num_item_types=5,
        max_employees=3,
        initial_employees=2,
        episode_length=100,
        render_mode=None
    )
    
    # Manually add some items to storage for testing
    grid = env.warehouse_grid
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.cell_types[y, x] == CellType.STORAGE.value and grid.item_grid[y, x] == -1:
                # Add item type 0 to first few empty storage cells
                if len([pos for pos in grid.storage_positions if grid.get_item_at_position(pos[0], pos[1]) == 0]) < 3:
                    grid.item_grid[y, x] = 0
                elif len([pos for pos in grid.storage_positions if grid.get_item_at_position(pos[0], pos[1]) == 1]) < 3:
                    grid.item_grid[y, x] = 1
                elif len([pos for pos in grid.storage_positions if grid.get_item_at_position(pos[0], pos[1]) == 2]) < 3:
                    grid.item_grid[y, x] = 2
    
    return env

def print_grid_state(env):
    """Print the current state of the warehouse grid"""
    print("\n=== WAREHOUSE STATE ===")
    print("Grid layout (C=corridor, S=storage with item, E=empty storage, P=packing, Z=spawn):")
    
    grid = env.warehouse_grid
    employee_positions = {emp.position: f"A{emp.id}" for emp in env.employees}
    
    for y in range(grid.height):
        row = ""
        for x in range(grid.width):
            pos = (x, y)
            if pos in employee_positions:
                row += f"{employee_positions[pos]:>3}"
            elif grid.cell_types[y, x] == CellType.STORAGE.value:
                item = grid.item_grid[y, x]
                if item != -1:
                    row += f"S{item:>2}"
                else:
                    row += "  E"
            elif grid.cell_types[y, x] == CellType.PACKING_STATION.value:
                row += "  P"
            elif grid.cell_types[y, x] == CellType.SPAWN_ZONE.value:
                row += "  Z"
            elif grid.cell_types[y, x] == CellType.EMPTY.value:
                row += "  ."
            else:
                row += "  ?"
            row += " "
        print(f"{y:2d}: {row}")
    
    print("\nEmployee details:")
    for emp in env.employees:
        print(f"  Agent {emp.id}: pos={emp.position}, state={emp.state.name}, "
              f"order={emp.current_order_id}, target={emp.target_position}")
        if emp.current_order_id:
            print(f"    Order items: {emp.order_items}, collected: {emp.items_collected}")
        if emp.path:
            print(f"    Path: {emp.path[:3]}..." if len(emp.path) > 3 else f"    Path: {emp.path}")
        if hasattr(emp, 'collision_wait_count') and emp.collision_wait_count > 0:
            print(f"    Collision wait: {emp.collision_wait_count}")
        if hasattr(emp, 'stuck_timer') and emp.stuck_timer > 0:
            print(f"    Stuck timer: {emp.stuck_timer}")

def test_basic_movement():
    """Test basic agent movement and pathfinding"""
    print("=== TESTING BASIC MOVEMENT ===")
    
    env = create_simple_test_env()
    obs, info = env.reset()
    
    print("Initial state:")
    print_grid_state(env)
    
    # Create a manual order for testing
    from environment.order_generator import Order
    
    # Find what items are available in storage
    available_items = []
    for y in range(env.warehouse_grid.height):
        for x in range(env.warehouse_grid.width):
            if env.warehouse_grid.cell_types[y, x] == CellType.STORAGE.value:
                item = env.warehouse_grid.item_grid[y, x]
                if item != -1 and item not in available_items:
                    available_items.append(item)
    
    print(f"Available items in storage: {available_items}")
    
    if available_items:
        # Create simple order with first available item
        test_order = Order(id=999, items=[available_items[0]], value=100.0, arrival_time=0)
        env.order_queue.add_order(test_order)
        print(f"Created test order: ID={test_order.id}, items={test_order.items}")
        
        # Manually assign order to first idle employee
        idle_emp = None
        for emp in env.employees:
            if emp.state == EmployeeState.IDLE and not emp.is_manager:
                idle_emp = emp
                break
        
        if idle_emp:
            print(f"Assigning order to employee {idle_emp.id}")
            if idle_emp.set_order(test_order.id, test_order.items):
                test_order.claim()
                env._assign_employee_to_order(idle_emp, test_order)
                print(f"Order assigned successfully")
            else:
                print(f"Failed to assign order")
        else:
            print("No idle employees found")
    
    print("\nAfter order assignment:")
    print_grid_state(env)
    
    # Step simulation for several timesteps
    print(f"\n=== STEPPING SIMULATION ===")
    for step in range(20):
        print(f"\n--- Step {step} ---")
        
        # Take empty action (no agent commands)
        action = {
            'staffing_action': 0,
            'layout_swap': [0, 0], 
            'order_assignments': [0] * 20
        }
        
        obs, reward, done, truncated, info = env.step(action)
        
        print_grid_state(env)
        
        # Check if any agents are stuck
        stuck_agents = []
        for emp in env.employees:
            if hasattr(emp, 'stuck_timer') and emp.stuck_timer > 2:
                stuck_agents.append(emp.id)
            if hasattr(emp, 'collision_wait_count') and emp.collision_wait_count > 3:
                stuck_agents.append(emp.id)
        
        if stuck_agents:
            print(f"STUCK AGENTS DETECTED: {stuck_agents}")
            # Check if they're in front of storage
            for emp_id in stuck_agents:
                emp = next(e for e in env.employees if e.id == emp_id)
                x, y = emp.position
                print(f"  Agent {emp_id} stuck at {emp.position}")
                
                # Check adjacent cells
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    adj_x, adj_y = x + dx, y + dy
                    if env.warehouse_grid.is_valid_position(adj_x, adj_y):
                        cell_type = env.warehouse_grid.cell_types[adj_y, adj_x]
                        if cell_type == CellType.STORAGE.value:
                            item = env.warehouse_grid.item_grid[adj_y, adj_x]
                            print(f"    Adjacent storage at ({adj_x}, {adj_y}) has item {item}")
                
                # Check target item position
                if hasattr(emp, 'target_item_position') and emp.target_item_position:
                    print(f"    Target item position: {emp.target_item_position}")
        
        if done or truncated:
            break
        
        # Stop if all agents are idle (completed tasks)
        all_idle = all(emp.state == EmployeeState.IDLE for emp in env.employees)
        if all_idle:
            print("All agents idle - test complete")
            break

def test_picking_logic():
    """Test the picking logic specifically"""
    print("\n=== TESTING PICKING LOGIC ===")
    
    env = create_simple_test_env()
    obs, info = env.reset()
    
    # Find a storage cell with an item
    storage_pos = None
    item_type = None
    for y in range(env.warehouse_grid.height):
        for x in range(env.warehouse_grid.width):
            if (env.warehouse_grid.cell_types[y, x] == CellType.STORAGE.value and
                env.warehouse_grid.item_grid[y, x] != -1):
                storage_pos = (x, y)
                item_type = env.warehouse_grid.item_grid[y, x]
                break
        if storage_pos:
            break
    
    if not storage_pos:
        print("No storage with items found!")
        return
    
    print(f"Found storage at {storage_pos} with item {item_type}")
    
    # Find adjacent walkable position
    adjacent_pos = None
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        adj_x, adj_y = storage_pos[0] + dx, storage_pos[1] + dy
        if env.warehouse_grid.is_walkable(adj_x, adj_y):
            adjacent_pos = (adj_x, adj_y)
            break
    
    if not adjacent_pos:
        print("No adjacent walkable position found!")
        return
    
    print(f"Adjacent walkable position: {adjacent_pos}")
    
    # Place an employee at the adjacent position
    emp = env.employees[0]
    emp.position = adjacent_pos
    emp.state = EmployeeState.MOVING
    emp.current_order_id = 999
    emp.order_items = [item_type]
    emp.items_collected = []
    emp.target_position = None
    emp.path = []
    
    print(f"Placed employee {emp.id} at {adjacent_pos}")
    print("Employee should detect adjacent item and start picking...")
    
    # Step a few times to see picking behavior
    for step in range(8):
        print(f"\n--- Picking Step {step} ---")
        
        # Simulate employee step
        assigned_positions = {e.position for e in env.employees if e != emp}
        result = emp.step(env.warehouse_grid, assigned_positions)
        
        print(f"Step result: {result}")
        print(f"Employee state: {emp.state.name}")
        print(f"Employee position: {emp.position}")
        print(f"Target position: {emp.target_position}")
        if hasattr(emp, 'target_item_position'):
            print(f"Target item position: {emp.target_item_position}")
        print(f"Items collected: {emp.items_collected}")
        print(f"Items needed: {emp.order_items}")
        print(f"Task timer: {emp.task_timer}")
        
        # Check logic
        items_needed = set(emp.order_items)
        items_collected = set(emp.items_collected)
        print(f"Set comparison: needed={items_needed}, collected={items_collected}")
        print(f"All collected? {items_needed.issubset(items_collected)}")
        
        if emp.state == EmployeeState.IDLE:
            print("Employee completed task!")
            break
        elif emp.state == EmployeeState.DELIVERING:
            if len(emp.items_collected) < len(emp.order_items):
                print("BUG: Employee going to delivery without all items!")
            else:
                print("Employee going to delivery with all items!")
            break

def test_simple_case():
    """Test the most basic case: agent next to item needs to pick it"""
    print("\n=== TESTING SIMPLE ADJACENT PICKING ===")
    
    # Create minimal setup
    env = create_simple_test_env()
    obs, info = env.reset()
    
    # Find a good test location  
    storage_pos = (12, 1)  # Known storage location from the grid printout
    adjacent_pos = (5, 5)  # Far away position to test pathfinding
    item_type = 0  # Item type 0 should be there
    
    print(f"Testing with storage at {storage_pos}, adjacent walkable at {adjacent_pos}")
    print(f"Item type at storage: {env.warehouse_grid.get_item_at_position(storage_pos[0], storage_pos[1])}")
    print(f"Is adjacent walkable: {env.warehouse_grid.is_walkable(adjacent_pos[0], adjacent_pos[1])}")
    
    # Setup employee
    emp = env.employees[0]
    emp.position = adjacent_pos
    emp.state = EmployeeState.MOVING
    emp.current_order_id = 999
    emp.order_items = [item_type, 1]  # Need both item 0 and item 1
    emp.items_collected = []
    emp.target_position = None
    emp.path = []
    emp.task_timer = 0
    
    # Test step by step
    for step in range(15):  # More steps to handle multiple items
        print(f"\n--- Step {step} ---")
        print(f"Before: state={emp.state.name}, pos={emp.position}, target={emp.target_position}")
        
        assigned_positions = {e.position for e in env.employees if e != emp}
        result = emp.step(env.warehouse_grid, assigned_positions)
        
        print(f"After: state={emp.state.name}, pos={emp.position}, target={emp.target_position}")
        print(f"Result: {result}")
        print(f"Items collected: {emp.items_collected}")
        print(f"Items needed: {emp.order_items}")
        
        if result.get('picked_item'):
            print(f"SUCCESS: Picked item {result['picked_item']}")
        
        if emp.state == EmployeeState.DELIVERING:
            if len(emp.items_collected) < len(emp.order_items):
                print("BUG: Going to delivery without all items!")
            else:
                print("Going to delivery with all items - test passed!")
            break
        elif emp.state == EmployeeState.MOVING and len(emp.items_collected) > 0:
            print(f"Good: Collected some items, continuing to find more...")
            # Continue the loop to collect more items

if __name__ == "__main__":
    print("Starting collision and state management debug tests...")
    
    try:
        test_simple_case()
        test_picking_logic()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDebug tests completed.")