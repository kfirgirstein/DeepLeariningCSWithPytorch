r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=8,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=8,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

When using an appropriate baseline which is the mean of all the q_vals, this would simply reduce the variance of the policy gradient. Reducing the mean of the q_vals would center the values around 0. Mathematically it can be easily shown the centering the values around the 0 instead high values for example will reduce the variance. 
Reducing the variance makes our model stricter! This way actions which can be called more 'adventures' and might cause a poor behavior would have less chance to be taken.  
"""


part1_q2 = r"""
**Your answer:**


$v_\pi(s)$ sort of tries to predict the value of the average action over all possible actions, where $q_\pi(s,a)$ is the expected value of a specific action. This how we get the advantage which tells us how much better or worse an action is respectively to the average action value. 

"""


part1_q3 = r"""
**Your answer:**


1. As can be seen we got the best performance with BPG and CPG. The reason for that is that when using entropy loss, the model becomes more complex and explores more actions. As a result, the model will use a more complex combinations of actions instead of few actions which he thinks they are the best at the beginning and therefore tries to maximize his performance with this set of few actions.
Another interesting thing we can is that we have achieved better performance with baseline. This we contribute to the fact that as we explained in answer 1 a stricter model would perform better in this game environment.
2. As we can see it takes more time for the AAC model to learn, this can be easily seen on the mean reward graph. The learning rate at the beginning of the CPG model is much steeper. However, at the long term in the AAC model the mean reward gets very close to the mean reward of the of the CPG model. This makes us believe that the AAC model might have a better potential for learning than the CPG.
Additionally, the delta in the entropy loss is much higher for the AAC and even at the end the loss is bigger than the CPG entropy loss. This is also why we also believe that the AAC can become a better model by having more complex action combinations after it will learn more.


"""
