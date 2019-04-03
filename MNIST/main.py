import tensorflow as tf
import pygame
import pygame.freetype  # Import the freetype module.
import time

if __name__ == "__main__":
    # Load data into a test and validation set
    mnist = tf.keras.datasets.mnist
    (training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()
    # Normalize data
    training_inputs, test_inputs = training_inputs / 255.0, test_inputs / 255.0

    # Create the neural network
    # This was a model used in an example that was found while researching the topology for this problem,
    # with this model we were able to score 98~% on the validation set
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(training_inputs, training_labels, epochs=2)

    # Validate teh model (98% avg @:2 epocs)
    model.evaluate(test_inputs, test_labels)

    # Use the network / Render the numbers
    pygame.init()
    pygame.font.init()
    myfont = pygame.freetype.SysFont('Comic Sans MS', 30)
    tile_size = 20
    screen = pygame.display.set_mode((tile_size * 28, tile_size * 28))

    running = True
    index = 0
    predictions = model.predict(test_inputs)
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw number
            test_input = test_inputs[index]
            test_label = test_labels[index]
            prediction = predictions[index]
            for i in range(len(test_input)):
                row = test_input[i]
                for j in range(len(row)):
                    pixel = test_input[j][i]
                    color = (pixel * 255, pixel * 255, pixel * 255)
                    pygame.draw.rect(screen, color, pygame.Rect(i * tile_size, j * tile_size, tile_size, tile_size))

            index += 1
            index %= len(test_inputs)
            # Draw label of the number and what the model predicted
            myfont.render_to(screen, (5, 5), str(test_label) + ' =', (255, 255, 255))
            myfont.render_to(screen, (50, 5), str(prediction.argmax()), (255, 255, 255))

            # Display everything
            pygame.display.flip()

            # Small delay
            time.sleep(1)
        pygame.quit()
    except SystemExit:
        pygame.quit()
