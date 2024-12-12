from gymnasium.envs.registration import register

register(
    id="Deng_gym_cutting_stock/CuttingStock-v0",
    entry_point="deng_gym_cutting_stock.envs:DengCuttingStockEnv",
)
