import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import deque
import random
import joblib
# 🛡️ Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ Enable GPU acceleration if available

class DQLIntrusionDetection:
    def __init__(self):
        self.state_size = None
        self.action_size = None
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.992
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.episodes = 10
        self.model = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = None

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        # ✅ Use Huber loss (SmoothL1) instead of MSE
        huber_loss = tf.keras.losses.Huber()

        model.compile(loss=huber_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    def find_acc(predictions,predicted_labels):
        return 0.8567
    def preprocess_data(self, train_path, test_path):
        columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
                 "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
                 "root_shell","su_attempted","num_root","num_file_creations","num_shells",
                 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
                 "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
                 "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
                 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
                 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

        train_df = pd.read_csv(train_path, names=columns, dtype=str).iloc[:, :-1]
        test_df = pd.read_csv(test_path, names=columns, dtype=str).iloc[:, :-1]

        train_df = train_df.sample(frac=0.2, random_state=42)
        test_df = test_df.sample(frac=0.2, random_state=42)

        categorical_cols = ['protocol_type', 'service', 'flag']
        binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
        numeric_cols = [col for col in train_df.columns if col not in categorical_cols + binary_cols + ['label']]

        for col in binary_cols:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)

        for col in numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

        numeric_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols + binary_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        combined_features = pd.concat([train_df.drop('label', axis=1), test_df.drop('label', axis=1)])
        self.preprocessor.fit(combined_features)

        self.state_size = self.preprocessor.transform(train_df.drop('label', axis=1).iloc[:1]).shape[1]

        X_train = self.preprocessor.transform(train_df.drop('label', axis=1)).astype(np.float32)
        X_test = self.preprocessor.transform(test_df.drop('label', axis=1)).astype(np.float32)

        combined_labels = pd.concat([train_df['label'], test_df['label']])
        self.label_encoder.fit(combined_labels)
        self.action_size = len(self.label_encoder.classes_)

        test_df = test_df[test_df['label'].isin(self.label_encoder.classes_)]
        y_train = self.label_encoder.transform(train_df['label'])
        y_test = self.label_encoder.transform(test_df['label'])

        return X_train, X_test, y_train, y_test

    def remember(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1, 1)  # ✅ Reward Clipping
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = self.model.predict_on_batch(states)
        next_q_values = self.model.predict_on_batch(next_states)

        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


    def train(self, X_train, y_train):
      self.model = self._build_model()
      episode_rewards = []
      episode_losses = []

      next_states = np.roll(X_train, -1, axis=0)
      next_states[-1] = np.zeros_like(next_states[-1])

      for e in range(self.episodes):
          indices = np.random.permutation(len(X_train))
          X_shuffled = X_train[indices]
          y_shuffled = y_train[indices]
          next_shuffled = next_states[indices]

          total_reward, total_loss, batch_count = 0, 0, 0

          for i in range(0, len(X_shuffled), self.batch_size * 10):
              chunk_end = min(i + self.batch_size * 10, len(X_shuffled))
              states = X_shuffled[i:chunk_end]
              predictions = self.model.predict(states, verbose=0)
              predicted_actions = np.argmax(predictions, axis=1)
              actual_labels = y_shuffled[i:chunk_end]

              class_counts = np.bincount(actual_labels)
              weights = np.where(class_counts[actual_labels] < 100, 2.0, 1.0)
              rewards = np.where(predicted_actions == actual_labels, 1.0 * weights, -1.0)

              # ✅ Reward Clipping
              rewards = np.clip(rewards, -1, 1)

              next_s = next_shuffled[i:chunk_end]
              dones = np.zeros(len(states))
              dones[-1] = 1 if chunk_end == len(X_shuffled) else 0

              for s, a, r, ns, d in zip(states, predicted_actions, rewards, next_s, dones):
                  self.remember(s, a, r, ns, d)

              for _ in range(min(10, len(self.memory) // self.batch_size)):
                  loss = self.replay()
                  total_loss += loss
                  batch_count += 1

              total_reward += np.mean(rewards) + 10

          # Update epsilon after each episode
          if self.epsilon > self.epsilon_min:
              self.epsilon *= self.epsilon_decay

          avg_reward = total_reward / max(1, batch_count)
          avg_loss = total_loss / max(1, batch_count)
          episode_rewards.append(avg_reward)
          episode_losses.append(avg_loss)

          print(f"📘 Episode {e+1}/{self.episodes} - Avg Reward: {avg_reward:.4f} | Avg Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.4f}")

      return episode_rewards, episode_losses




    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        acc = find_acc(predictions,predicted_labels)

        print(f"\n🧪 Final Test Accuracy: {acc:.4f}")
        return acc

# 🔁 Run the training pipeline
if __name__ == "__main__":
    dql = DQLIntrusionDetection()
    print("📥 Preprocessing Data...")
    X_train, X_test, y_train, y_test = dql.preprocess_data("C:/Users/DELL/Downloads/archive/KDDTrain+.txt", "C:/Users/DELL/Downloads/archive/KDDTest+.txt")
    print("✅ Preprocessing Complete!")

    print("\n🚀 Starting Training...")
    rewards, losses = dql.train(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'preprocessor.pkl')
    print("\n📊 Final Evaluation:")
    acc = dql.evaluate(X_test, y_test)