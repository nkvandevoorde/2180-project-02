%% HELPER METHODS %%

%Compute error, false negative, and false positive rate (4, 5, & 7)
function Comparisons = label_comparisons(predicted_label, true_label)
    %Compare Labels
    true_positives = sum(predicted_label == 1 & true_label == 1);
    true_negatives = sum(predicted_label == 0 & true_label == 0);
    false_postives = sum(predicted_label == 1 & true_label == 0);
    false_negatives = sum(predicted_label == 0 & true_label == 1);

    %Compute Comparisons
    true_allocations = length(true_label);
    Comparisons.ER = (false_postives + false_negatives)/true_allocations;
    Comparisons.FPR = false_postives/(false_postives + true_negatives);
    Comparisons.FNR = false_negatives/(false_negatives + true_positives);

    %Display Comparisons
    disp('Error Rate: ' + Comparisons.ER)
    disp('False Positive Rate: ' + Comparisons.FPR)
    disp('False Negative Rate: ' + Comparisons.FNR)

    %For M values comparisons
    %Display the comparison values for each M in a table
    if exists ('M_Values','vars') && ~isempty(M_Values)
        ComparisonsTable = table(M_Values(:), Comparisons.ER(:), ...
            Comparisons.FPR(:), Comparisons.FNR(:), ...
            'VariableNames', {'M', 'Error Rate', 'False Positive Rate', ...
                'False Negative Rate'})
        %Display comparisons M = 20,...,5000
        disp('----Comparison Table----')
        disp(ComparisonsTable);

        %Plot to compare Error rates
        figure;
        plot(M_values, Comparisons.ER, '-o', 'LineWidth', 1.5);
        xlabel('M (Number of Hidden Nodes)');
        ylabel('Error Rate');
        title('Error Rate vs M');
        grid on;
    end
end

%create matrix of random values
function R = random_feature_matrix(M, M0)
    R = sign(rand(M, M0));
end

%Cluster via new features 
function Random_Features = compute_random_features(X, R)
   %X = new samples x old features
   %R = projection matrix
   %Z = Transformed features

    Z = max(0, X * R');
end

%Training function 
function Features = Least_squares_training(Z, Y)
    theta = inv(Z' * Z) * Z' * y;
end