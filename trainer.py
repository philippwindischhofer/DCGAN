import numpy as np

class gan_trainer:
    def __init__(self, disc, gen, adv, training_data):
        self.disc = disc
        self.adv = adv
        self.gen = gen
        self.training_data = training_data

        self.halfbatch_size = 64

    def train(self, number_epochs, epoch_callback):
        training_dataset_length = np.shape(self.training_data)[0]
        
        for epoch in range(number_epochs):
            print("============================================================")
            print(" batch {}".format(epoch))
            print("============================================================")
            
            # first, train the discriminator in a standalone way

            # prepare discriminator training data
            inds = np.random.randint(0, training_dataset_length, size = self.halfbatch_size)
            samples = self.training_data[inds]
            samples_target = np.full(self.halfbatch_size, 1.0)

            noise = np.random.uniform(-1.0, 1.0, size = (self.halfbatch_size, 100))
            samples_gen = self.gen.predict(noise)
            samples_gen_target = np.full(self.halfbatch_size, 0.0)

            training_in = np.concatenate([samples, samples_gen])
            training_target = np.concatenate([samples_target, samples_gen_target])

            loss, acc = self.disc.train_on_batch(training_in, training_target)
            print("discriminator: loss = {}, acc = {}".format(loss, acc))
            
            # then, train the full adversarial pair with frozen discriminator weights: noise as input, 1.0 as target output
            training_in = np.random.uniform(-1.0, 1.0, size = (self.halfbatch_size, 100))
            training_target = np.full(self.halfbatch_size, 1.0)

            loss, acc = self.adv.train_on_batch(training_in, training_target)
            print("adversarial pair: loss = {}, acc = {}".format(loss, acc))

            epoch_callback(self.gen, epoch)
