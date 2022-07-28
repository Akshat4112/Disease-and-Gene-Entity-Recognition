import nlu

pipe = nlu.load(biobert)

pipe.predict("He is amazing")

