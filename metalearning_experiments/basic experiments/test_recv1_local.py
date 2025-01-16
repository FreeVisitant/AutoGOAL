from sklearn.datasets import fetch_rcv1

rcv1 = fetch_rcv1(
    subset='train',
    download_if_missing=True, 
    data_home='/root/scikit_learn_data'  #
)
print("RCV1 shape:", rcv1.data.shape)
print("Labels shape:", rcv1.target.shape)
print("OK - Train loaded offline, and .pkl created.")
