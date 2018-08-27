import numpy as np

class gan_trainer:
    def __init__(self, disc, gen, adv, training_data):
        self.disc = disc
        self.adv = adv
        self.gen = gen
        self.training_data = training_data

        self.halfbatch_size = 64

    def train(self, number_batches, batch_callback):
        training_dataset_length = np.shape(self.training_data)[0]
        print(np.shape(self.training_data))
        
        for batch in range(number_batches):            
            # first, train the discriminator in a standalone way

            print("============================================================")
            print(" batch {}".format(batch))
             
            # prepare discriminator training data
            inds = np.random.randint(0, training_dataset_length, size = self.halfbatch_size)
            print(" using samples {}".format(str(inds)))

            # true samples: target == 0.0
            samples = self.training_data[inds]
            samples_target = np.random.uniform(0.0, 0.1, self.halfbatch_size)

            # generated samples: target == 1.0
            noise = np.random.uniform(-1.0, 1.0, size = (self.halfbatch_size, 100))
            samples_gen = self.gen.predict(noise)
            samples_gen_target = np.random.uniform(0.9, 1.0, self.halfbatch_size)

            training_in = np.concatenate([samples, samples_gen])
            training_target = np.concatenate([samples_target, samples_gen_target])

            loss, acc = self.disc.train_on_batch(training_in, training_target)
            print("discriminator: loss = {}, acc = {}".format(loss, acc))
            
            # then, train the full adversarial pair with frozen discriminator weights: noise as input, 1.0 as target output
            training_in = np.random.uniform(-1.0, 1.0, size = (self.halfbatch_size, 100))
            training_target = np.full(self.halfbatch_size, 0.0)

            loss, acc = self.adv.train_on_batch(training_in, training_target)
            print("adversarial pair: loss = {}, acc = {}".format(loss, acc))

            batch_callback(self.gen, batch)

            print("============================================================")
