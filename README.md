# TPT_Translation

To train this model in a distributed environment, run the following from the root folder of the repository:

```
accelerate launch model/model.py [train/train_tph]
```

To evaluate the models, run:

```
python model/test.py
```

a lot of things that should be command-line options are kinda hard-coded in and there's a lot of really bad spaghetti code i'm sorry i really had to rush maybe i can fix that in the future :groeL;
