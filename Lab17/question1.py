"""Let x1 = [3, 6], x2 = [10, 10]. Use the above “Transform” function to transform these
vectors to a higher dimension and compute the dot product in a higher dimension. Print
the value."""
import numpy as np

def transform(x):

    # feature mapping function
    x1,x2=x
    return np.array([ x1**2,
        np.sqrt(2)*x1*x2,
        x2**2
    ])

def compute_dot_product(phi1,phi2):
    dot_product = np.dot(phi1,phi2)
    return dot_product


def main():
    x1 = np.array([3, 6])
    x2 = np.array([10, 10])
    phi1=transform(x1)
    phi2=transform(x2)
    print("Transformed value of x1:",phi1)
    print("Transformed value of x2:",phi2)

    #dot_product
    dot_product = compute_dot_product(phi1,phi2)
    print("Dot product of x1 and x2:",dot_product)


if __name__ == "__main__":
    main()



#
# transformed_data = []
# for i,j in x1,x2:
#     X_transformed = transform(i,j)
#     transformed_data.append(X_transformed)
# print(np.array(transformed_data))

