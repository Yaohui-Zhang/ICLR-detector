import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
class MLE():
    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        parquet_file_path = os.path.join(current_dir, 'Parameters.parquet')
        sorted_df=pd.read_parquet(parquet_file_path)
        self.all_tokens = sorted_df['Word'].tolist()
        self.all_tokens_set = set(self.all_tokens)
        self.log_p_hat = {row['Word']: row['Human Log Probability'] for index, row in sorted_df.iterrows()}
        self.log_q_hat = {row['Word']: row['AI Log Probability'] for index, row in sorted_df.iterrows()}
        self.log_p_precomputed = {t: np.log1p(-np.exp(self.log_p_hat.get(t, -13.8))) for t in self.all_tokens}
        self.log_q_precomputed = {t: np.log1p(-np.exp(self.log_q_hat.get(t, -13.8))) for t in self.all_tokens}
    def optimized_log_likelihood(self,alpha, log_p_values, log_q_values):
        alpha = alpha[0]
        avg_ll = np.mean(np.log((1 - alpha)  + alpha * np.exp(log_q_values-log_p_values)))
        return -avg_ll
    @staticmethod
    def process_single(path,already_explode=False):
        df_sampled = pd.read_parquet(path)
        if already_explode:
            df_sampled.dropna(subset=['inference_sentence'], inplace=True)
            df_sampled=df_sampled[df_sampled['inference_sentence'].apply(len) > 10]
            return df_sampled
        df_sampled['inference_sentence']=df_sampled['inference_sentence'].apply(lambda x:[inner_array.tolist() for inner_array in x])
        df_sampled=df_sampled.explode('inference_sentence')
        df_sampled.dropna(subset=['inference_sentence'], inplace=True)
        df_sampled=df_sampled[df_sampled['inference_sentence'].apply(len) > 10]
        return df_sampled
     
    def precompute_log_probabilities(self,data):

        # Vectorization of log_P and log_Q calculations
        log_p_values = [sum(self.log_p_hat.get(t, -13.8) for t in x) + 
                        sum(self.log_p_precomputed[t] for t in self.all_tokens if t not in x) for x in data]
        log_q_values = [sum(self.log_q_hat.get(t, -13.8) for t in x) + 
                        sum(self.log_q_precomputed[t] for t in self.all_tokens if t not in x) for x in data]
        return np.array(log_p_values), np.array(log_q_values)
    
    def bootstrap_alpha_inference(self,data,n_bootstrap=1000):
        full_log_p_values, full_log_q_values = self.precompute_log_probabilities(data)
        alpha_values_bootstrap = []
        for i in range(n_bootstrap):
            sample_indices = np.random.choice(len(data), size=len(data), replace=True)
            sample_log_p_values = full_log_p_values[sample_indices]
            sample_log_q_values = full_log_q_values[sample_indices]
            result = minimize(self.optimized_log_likelihood, x0=[0.5], args=(sample_log_p_values, sample_log_q_values),
                            method='L-BFGS-B', bounds=[(0, 1)])
            if result.success:
                min_loss_alpha = result.x[0]
                alpha_values_bootstrap.append(min_loss_alpha)
        return np.percentile(alpha_values_bootstrap, [2.5, 97.5])

    def inference(self,reviews_path,already_explode=False):
        final=MLE.process_single(reviews_path,already_explode=already_explode)
        data = final['inference_sentence'].apply(lambda x: set(token for token in x if token in self.all_tokens_set))
        confidence_interval=self.bootstrap_alpha_inference(data)
        solution=round(np.mean(confidence_interval), 3)
        half_width = (confidence_interval[1] - confidence_interval[0]) / 2
        half_width=round(half_width, 3)
        print(f"{'Prediction':>10},{'CI':>10}")
        print(f"{solution:10.3f},{half_width:10.3f}")
        return solution*100,half_width*100
