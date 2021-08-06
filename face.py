def global_face_prediction(img)
    import pickle
    from PIL import Image
    import io
    
    from mtcnn.mtcnn import MTCNN
    from numpy import asarray
    
    from numpy import expand_dims
    from keras.models import load_model
    
    model_file = 'homenikitaglobus1endpoint-model-testfacenet_keras.h5'
    filename = 'homenikitaglobus1endpoint-model-testfinalized_model.sav'
    #filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    try
    #model_file = 'facenet_keras.h5'
        model2 = load_model(model_file)
    except
        return JOPA (
    
    # extract a single face from a given photograph
    def extract_face(image, required_size=(160, 160))
        # load image from file
        #image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1y2, x1x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    
    # get the face embedding for one face
    def get_embedding(model, face_pixels)
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean)  std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]
    
    def predict_person(img)
        #img = Image.open(io.BytesIO(img))
        #script_dir = pathlib.Path(__file__).parent.absolute()
        #model_file = os.path.join(script_dir, 'facenet_keras.h5')


        img = img.resize((160, 160))

        face_pixels = extract_face(img)
        face_embedding = get_embedding(model2, face_pixels)

        # prediction for the face
        samples = expand_dims(face_embedding, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index]  100

        return class_index, class_probability, face_pixels
    
    class_index, class_probability, face_pixels = predict_person(img)
    return class_index, class_probability
            
            
from PIL import Image
test_image1 = Image.open("nikita_test_3.jpg")
print(global_face_prediction(test_image1))
