import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from tensorflow.keras import Model
import tensorflow as tf

class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb, userIDs, itemIDs):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu, dtype=tf.float32)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001), dtype=tf.float32)
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001), dtype=tf.float32)
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001), dtype=tf.float32)
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001), dtype=tf.float32)
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i] +\
            tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * tf.reduce_sum(self.betaU**2) +\
                           tf.reduce_sum(self.betaI**2) +\
                           tf.reduce_sum(self.gammaU**2) +\
                           tf.reduce_sum(self.gammaI**2)
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i +\
               tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)

def trainingStep(interactions, model, optimizer, userIDs, itemIDs):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,r = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)

        loss = model(sampleU,sampleI,sampleR)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()