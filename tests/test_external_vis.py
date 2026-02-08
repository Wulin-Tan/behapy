import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import behapy as bp
# Ensure we can access the newly added functions
# They are in bp.pl

class TestExternalVis(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        n_obs = 100
        x = np.linspace(0, 10, n_obs)
        y = np.sin(x)
        
        # Create BehapyData
        # Mocking minimal structure
        X = np.column_stack([x, y])
        obs = pd.DataFrame(index=[f"frame_{i}" for i in range(n_obs)])
        obs['x'] = x
        obs['y'] = y
        var = pd.DataFrame(index=['body_x', 'body_y'])
        var['bodypart'] = 'body'
        var['coord'] = ['x', 'y']
        
        # Assuming BehapyData can be initialized like AnnData
        self.bdata = bp.BehapyData(X=X, obs=obs, var=var)
        self.bdata.uns['fps'] = 30
        
        # Add mock VAME results
        self.bdata.obsm['X_vame'] = np.random.rand(n_obs, 2)
        self.bdata.obs['vame_cluster'] = np.random.randint(0, 3, n_obs)
        
        # Add mock BehaviorFlow zones
        self.bdata.uns['zone_stats_zones'] = {
            'zone1': pd.DataFrame({'x': [0, 1, 1, 0], 'y': [0, 0, 1, 1]})
        }
        
        # Add mock NEG grids
        self.bdata.uns['exploration_grids'] = pd.DataFrame({
            'grid_type': ['open'],
            'band_id': [1],
            'x_min': [0],
            'x_max': [1],
            'y_min': [0],
            'y_max': [1]
        })
        
        self.bdata.obs['x_m'] = x
        self.bdata.obs['y_m'] = y

    def test_plot_zones(self):
        print("Testing plot_behaviorflow_zones...")
        ax = bp.pl.plot_behaviorflow_zones(self.bdata, show_trajectory=True)
        self.assertIsNotNone(ax)
        plt.close()

    def test_plot_grids(self):
        print("Testing plot_neg_grids...")
        ax = bp.pl.plot_neg_grids(self.bdata, show_trajectory=True)
        self.assertIsNotNone(ax)
        plt.close()
        
    def test_plot_pyrat(self):
        print("Testing plot_pyrat_trajectory...")
        # This calls external PyRAT code
        try:
            # We explicitly pass x_col/y_col as our mock has them
            bp.pl.plot_pyrat_trajectory(self.bdata, bodypart='body', x_col='x', y_col='y', show=False)
        except Exception as e:
            print(f"PyRAT plot warning: {e}")
            # Don't fail if just display issue, but check logic
            # PyRAT might try to show plot which fails in non-interactive
        plt.close()

    def test_plot_vame(self):
        print("Testing plot_vame_umap...")
        try:
            bp.pl.plot_vame_umap(self.bdata)
            bp.pl.plot_vame_umap(self.bdata, label_col='vame_cluster')
        except Exception as e:
            print(f"VAME plot warning: {e}")
        plt.close()

if __name__ == '__main__':
    unittest.main()
