from Runner import Runner
from utils.GeneralQVisualizer import GeneralQVisualizer
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    # 打印当前使用的配置
    print(OmegaConf.to_yaml(cfg))
    
    runner = Runner(cfg)

    runner.visualizer = GeneralQVisualizer(
        runner.env, runner.device, 
        x_idx=0, y_idx=1, 
        x_range=(-0.5, 0.5), y_range=(0, 1.5),
        labels=("X Position (Horizontal)", "Y Position (Height)")
    )
    
    runner.run()

if __name__ == "__main__":
    main()