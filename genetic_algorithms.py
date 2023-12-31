import pygad
import pygad.kerasga
import tensorflow as tf
import tensorflow.keras


def fitness_func(ga_instanse, solution, sol_idx):
    global train_generator, data_outputs, keras_ga, model

    predictions = pygad.kerasga.predict(
        model=model, solution=solution, data=train_generator
    )

    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    solution_fitness = 1.0 / (cce(data_outputs, predictions).numpy() + 0.00000001)

    return solution_fitness


def on_generation(ga_instance):
    print(
        "Generation = {generation}".format(generation=ga_instance.generations_completed)
    )
    print(
        "Fitness    = {fitness}".format(
            fitness=ga_instance.best_solution(ga_instance.last_generation_fitness)[1]
        )
    )


# The dataset path.
dataset_path = r"C:\Users\rober\Downloads\flowers\flowers"
num_classes = 5
img_size = 150

# Create a simple CNN.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(img_size, img_size, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

data_generator = tf.keras.preprocessing.image.ImageDataGenerator()
train_generator = data_generator.flow_from_directory(
    dataset_path,
    class_mode="categorical",
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,
)
# train_generator.class_indices
data_outputs = tf.keras.utils.to_categorical(train_generator.labels)


initial_population = (
    keras_ga.population_weights
)  # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(
    num_generations=20,
    num_parents_mating=5,
    initial_population=initial_population,
    fitness_func=fitness_func,
    on_generation=on_generation,
)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(
    ga_instance.last_generation_fitness
)
print(
    "Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness
    )
)
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

predictions = pygad.kerasga.predict(
    model=model, solution=solution, data=train_generator
)
print("Predictions : \n", predictions)

# Calculate the categorical crossentropy for the trained model.
cce = tensorflow.keras.losses.CategoricalCrossentropy()
print("Categorical Crossentropy : ", cce(data_outputs, predictions).numpy())

# Calculate the classification accuracy for the trained model.
ca = tensorflow.keras.metrics.CategoricalAccuracy()
ca.update_state(data_outputs, predictions)
accuracy = ca.result().numpy()
print("Accuracy : ", accuracy)
