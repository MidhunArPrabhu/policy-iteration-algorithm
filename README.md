# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.


## POLICY IMPROVEMENT FUNCTION
#### Name:- MIDHUN AZHAHU RAJA P
#### Register Number:- 212222240066
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = np.argmax(Q, axis=1)

    return new_pi

def callable_policy(pi_array):
    return lambda s: pi_array[s]

pi_2_array = policy_improvement(V1, P)
pi_2 = callable_policy(pi_2_array)

print("Name:  MIDHUN AZHAHU RAJA P   ")
print("Register Number:212222240066      ")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)


```
## POLICY ITERATION FUNCTION
#### Name:- MIDHUN AZHAHU RAJA P
#### Register Number:- 212222240066
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    # Write your code here for policy iteration
    n_states = len(P)
    n_actions = len(P[0])

    # Initialize a random policy
    pi = np.random.randint(0, n_actions, n_states)

    while True:
        # Policy Evaluation
        V = np.zeros(n_states, dtype=np.float64)
        while True:
            prev_V = V.copy()
            for s in range(n_states):
                v = 0
                action = pi[s]
                for prob, next_state, reward, done in P[s][action]:
                    v += prob * (reward + gamma * prev_V[next_state] * (not done))
                V[s] = v
            if np.max(np.abs(prev_V - V)) < theta:
                break

        # Policy Improvement
        new_pi = np.zeros(n_states, dtype=np.int64)
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            new_pi[s] = np.argmax(Q[s])

        # Check for policy convergence
        if np.array_equal(pi, new_pi):
            break
        pi = new_pi

    return V, callable_policy(pi)
optimal_V, optimal_pi = policy_iteration(P)


print("Name:   MIDHUN AZHAHU RAJA P  ")
print("Register Number:  212222240066      ")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)






```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

#### Policy
<img width="723" height="168" alt="image" src="https://github.com/user-attachments/assets/c659fa9b-b518-49df-870f-56700140c6cd" />


#### Value function
<img width="741" height="166" alt="image" src="https://github.com/user-attachments/assets/ee653054-0f07-44fb-9aac-a0331fdbdbc6" />


#### success rate
<img width="707" height="44" alt="image" src="https://github.com/user-attachments/assets/3a75df58-9a91-4d67-bcf8-aaa62558ab00" />

</br>

### 2. Policy, Value function and success rate for the Improved Policy

#### Policy
<img width="566" height="161" alt="image" src="https://github.com/user-attachments/assets/81c03f0b-0c70-415f-9359-d016f9418f46" />

#### Value function
<img width="573" height="163" alt="image" src="https://github.com/user-attachments/assets/c7d1cf8b-53d6-4dc5-908f-a9c6e5ec0231" />


#### success rate
<img width="700" height="40" alt="image" src="https://github.com/user-attachments/assets/ef19fc8f-51a8-468e-8e88-dfdf44e6a4cf" />
</br>

### 3. Policy, Value function and success rate after policy iteration

</br>

#### Policy
<img width="899" height="195" alt="image" src="https://github.com/user-attachments/assets/7a45593c-741e-4d0a-a828-8fa52884e8f1" />

#### success rate
<img width="740" height="26" alt="image" src="https://github.com/user-attachments/assets/ef38791c-a579-4780-8e9a-1ed2cf8de5fc" />

#### Value function
<img width="925" height="139" alt="image" src="https://github.com/user-attachments/assets/b22a60aa-3e3b-4f91-8bf3-2c8d07a53587" />





## RESULT:

Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
