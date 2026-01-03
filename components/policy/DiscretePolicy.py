import torch

class DiscretePolicy:
    """
    职责：负责连接 Model 和 Action。
    它不包含探索逻辑，只负责根据 Model 的输出给出'理性'的动作。
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_action(self, state):
        """
        输入: state (numpy array)
        输出: action (int), q_values (tensor)
        """
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action, q_values