from src.watershed import WatershedEnv

def test_canCreateEnvironment():
    watershed = WatershedEnv()

def test_canRender():
    watershed = WatershedEnv()
    watershed.render()

def test_canRun():
    watershed = WatershedEnv()
    done = False
    while not done:
        observation, reward, done, info = watershed.step(0)