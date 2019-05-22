from kmean import init_centers_random, vect_avg, dist, get_nearest_center, init_centers_first_k, train_kmean, sum_of_within_group_ss
from data import load_data
import matplotlib.pyplot as plt 
import sys

# TODO: Please submit your plot and answers for Q2.4.1 - Q2.4.6 in your write up.
def main():
    val_names, data_set = load_data()
    iter_limit = 20

    # Q2.4.1: The values of sum of within group sum of squares for k = 5, k = 10 and k = 20.
    print("Q 2.4.1")
    for k in [5, 10, 20]:
        init_centers = init_centers_first_k(data_set, k)
        centers, clusters, num_iterations = train_kmean(data_set, init_centers, iter_limit)
        print( "k =", str(k) + ": " + str(sum_of_within_group_ss(clusters, centers)))
    print()

    # Q2.4.2: The number of iterations that k-means ran for k = 5.
    print ("Q 2.4.2")
    k = 5
    init_centers = init_centers_first_k(data_set, k)
    centers, clusters, num_iterations = train_kmean(data_set, init_centers, iter_limit)
    print ("k =", str(k) + ", num_iter: " + str(num_iterations))
    print()

    # Q2.4.3: A plot of the sum of within group sum of squares versus k for k = 1 - 50.
    # Please start your centers randomly (choose k points from the dataset at random).
    print ("Q 2.4.3")
    SSK = []
    min_ssk = 10000000
    min_ssk_c = 0

    # for k in range(1, 51):
    #     init_centers = init_centers_random(data_set, k)
    #     centers, clusters, num_iterations = train_kmean(data_set, init_centers, iter_limit)
    #     ssk = sum_of_within_group_ss(clusters, centers)
    #     print (str(k) + ", " + str(ssk))
    #     SSK.append(ssk)
    #     if ssk < min_ssk:
    #         min_ssk = ssk
    #         min_ssk_c = k
    # plt.plot([i+1 for i in range(50)], SSK, color='r')
    # plt.show()


    print ("Q 2.4.4")
    print("The best k is ", str(min_ssk_c), "because it has the minimun ssk = ", str(min_ssk))

    print("Q 2.4.5")
    init_centers = init_centers_random(data_set, 50)
    centers, clusters, num_iterations = train_kmean(data_set, init_centers, iter_limit)

    data_set_veg = vect_avg(data_set, 50)
    distances = []
    for i in range(50):
        dis = dist(centers[i], data_set_veg)
        print("the distance between ", str(i+1),"th center and avg of all countries is ", str(dis))
        distances.append(dis)

    plt.plot(distances)
    plt.show()

    plt.plot(distances,color='r')
    plt.ylim((0,10))
    plt.show()

    print("Q 2.4.6")
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            if clusters[i][j]['country'] == 'China':
                print("The country belongs to ", str(i), "the center")
                print("The set contain: ")
                for k in range(len(clusters[i])):
                    if k != j:
                        print(clusters[i][k]['country'])
                sys.exit(0)








    



if __name__ == "__main__":
    main()
