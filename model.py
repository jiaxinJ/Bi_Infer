class CMamba(nn.Module):
    def __init__(self):
        super(CMamba, self).__init__()
        self.conv = nn.Conv1d(136, 68, 1)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.pool2 = nn.AdaptiveAvgPool1d(32)
        self.mamba1 = Mamba(d_model=64, d_state=32, d_conv=3, expand=2)
        self.mamba2 = Mamba(d_model=32, d_state=32, d_conv=3, expand=2)
        self.ln = nn.Linear(32, 64)
    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.pool(x)
        x1 = self.mamba1(x)
        x2 = torch.permute(x, (0, 2, 1))
        x2 = self.conv2(x2)
        x2 = torch.permute(x2, (0, 2, 1))
        x2 = self.pool2(x2)
        x2 = self.mamba2(x2)
        x2 = self.ln(x2)
        x = x1 + x2
        return x

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.W_a = nn.Linear(feature_dim, feature_dim)
        self.W_b = nn.Linear(feature_dim, feature_dim)
        self.V = nn.Linear(feature_dim, 1)
    def forward(self, feature_a, feature_b):
        batch_size, num_features, feature_dim = feature_a.size()
        score_a = self.W_a(feature_a) 
        score_b = self.W_b(feature_b) 
        score = F.tanh(score_a + score_b) 
        attention_weights = F.softmax(self.V(score), dim=1) 
        weighted_sum = torch.bmm(attention_weights.permute(0, 2, 1), feature_b)  
        return weighted_sum  

class Feature_Fusion_Pre(nn.Module):
    def __init__(self):
        super(Feature_Fusion_Pre, self).__init__()
        self.attention_fusion = AttentionFusion(64)
        self.hidden_state = nn.Parameter(torch.randn(2 * 2, 1, 32))
        self.gru = torch.nn.GRU(input_size=64, hidden_size=32,
                          num_layers=2, dropout=0.5,
                          batch_first=True, bidirectional=True)
        self.pre_dropout = nn.Dropout(0.1)
        self.output = nn.Sigmoid()
    def forward(self, S, T):
        y = self.attention_fusion(S,T)
        y, _ = self.gru(y,self.hidden_state.repeat(1, y.shape[0], 1))  #[256, 1, 64]
        x = (S + T)/2
        x = torch.mean(x, dim=1)
        y = torch.reshape(y, (y.shape[0], y.shape[1]*y.shape[2]))   
        res = F.cosine_similarity(x, y, dim=1)
        res = self.pre_dropout(res)
        res = self.output(res)
        return res
