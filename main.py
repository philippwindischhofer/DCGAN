import imageio
import numpy as np
from pngreader import pngreader

from factory import factory
from trainer import gan_trainer

# remove alpha channel from png files
# convert -flatten img1.png img1-white.png

def save_sample(gen, epoch):
    # draw a few samples from the trained generator
    testinput = np.random.rand(1, 100)
    
    gen_output = gen.predict(testinput)

    gen_output = 127.5 * (gen_output + 1.0)
    
    print(np.shape(gen_output))
    
    # save the generated output images
    imageio.imwrite("trained_{}.png".format(epoch), gen_output[0])    

def main():
    training_data = pngreader.load_files("mnist/*.png")

    gen, disc, adv = factory.create_adversarial_pair()
    disc.summary()
    adv.summary()
    
    train = gan_trainer(disc, gen, adv, training_data)
    
    # perform the training for a small number of batches
    train.train(350, save_sample)

    print("done")

if __name__ == "__main__":
    main()
