# Check for required packages and install any that are missing
required_packages <- c(
  "tidyverse", "ggplot2", "ggExtra", "ggeffects", "lubridate", "ggpmisc", 
  "gridExtra", "grid", "dotwhisker", "lme4", "broom.mixed", "reshape2", 
  "stargazer", "dplyr", "fixest", "multiwayvcov", "cowplot", "emmeans", 
  "patchwork", "modelsummary", "kableExtra", "MASS", "sandwich", 
  "lmtest", "clubSandwich"
)

# Function to check and install missing packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages) > 0) {
    cat("Installing missing packages:", paste(new_packages, collapse=", "), "\n")
    install.packages(new_packages, repos="https://cloud.r-project.org")
  }
}

# Install missing packages
install_if_missing(required_packages)

# Load all packages, with error handling
for (package in required_packages) {
  tryCatch(
    {
      cat("Loading package:", package, "\n")
      library(package, character.only = TRUE)
    },
    error = function(e) {
      cat("Error loading", package, ":", conditionMessage(e), "\n")
      cat("Attempting to install and reload...\n")
      install.packages(package, repos="https://cloud.r-project.org")
      library(package, character.only = TRUE)
    }
  )
}

# Set working directory to project root (2 levels up from script location)
setwd(file.path(dirname(getwd())))

# Create output directories if they don't exist
dir.create("output", showWarnings = FALSE)
dir.create(file.path("output", "tables"), showWarnings = FALSE)
dir.create(file.path("output", "plots"), showWarnings = FALSE)

# Set up data -------------------------------------------------------------

# Read image and text data
data <- read_csv("output/complete_face_data.csv")
names(data)[names(data) == "biogudie_sim"] <- "face_bioguide"

# Drop duplicates
data <- distinct(data, face_id, image_name, filename, .keep_all = TRUE)

# Calculate number of missing images
num_rows_missing_images <- sum(is.na(data$angry))
print(num_rows_missing_images)

# Export dataframe with missing images
data_with_missing_images <- data[is.na(data$angry), ]
write.csv(data_with_missing_images, "data/missing_images_data.csv", row.names = FALSE)


# Modify the dataframe 'data' such that for rows where 'message' is NA,
# all variables starting with 'roberta_' are also set to NA.
data <- data %>%
  mutate(across(starts_with("roberta_"), ~ifelse(is.na(message), NA, .)))

# Scale emotion variables to range between 0 and 1
columns_to_scale <- c("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
data[columns_to_scale] <- data[columns_to_scale] / 100

# Rename happy to happiness
names(data)[names(data) == "happy"] <- "happiness"

# Read bioguide covariates data
load('data/bioguide_covariates.Rda')
bio_vars <- read_csv("data/bioguide_cross_base.csv")
bio_covs <- left_join(bio_covs, bio_vars)

# Read time-varying covariates data
load('data/time_covariates.Rda')

# merge and process race data
race_data <- read.csv('data/cel_race_bioguide.csv')
race_data <- race_data %>% rename('face_bioguide' = 'bioguide',
                                  'face_race_aa' = 'race_aa',
                                  'face_race_la' = 'race_la')

race_data <- race_data %>% 
    add_row(face_race_aa = 0, face_race_la = 1, face_bioguide = 'G000061') %>% 
    add_row(face_race_aa = 0, face_race_la = 0, face_bioguide = 'K000377') %>% 
    distinct()

data_merged <- merge(data, race_data, by="face_bioguide", all.x=TRUE)

# Generate non_neutral face display
data_merged$non_neutral <- 1 - data_merged$neutral

# Rename max emotion face display variable
names(data)[names(data) == "dominant_emotion"] <- "max_emo"

# Generate dates 
data_merged$datetime <- ymd_hms(data_merged$date)
data_merged$year <- year(data_merged$datetime)
data_merged$day <- day(data_merged$datetime)
data_merged$month <- month(data_merged$datetime)
data_merged$week <- week(data_merged$datetime)

# Merge bioguide covariates onto user bioguide
user_bio_covs <- bio_covs %>% 
    rename(user_bioguide = bioguide,
           user_age = age,
           user_medianIncomeE = medianIncomeE,
           user_dpres = dpres,
           user_net_trump_vote = net_trump_vote,
           user_followers = followers,
           user_nokken_poole_dim1 = nokken_poole_dim1,
           user_nokken_poole_dim2 = nokken_poole_dim2,
           user_margin_pc = margin_pc,
           user_last_name = last_name,
           user_first_name = first_name,
           user_type = type,
           user_gender = gender,
           user_state = state,
           user_district = district,
           user_party = party,
           user_birthday = birthday)

data_merged <- left_join(data_merged, user_bio_covs)

# Merge bioguide covariates onto face bioguide
face_bio_covs <- bio_covs %>% 
    rename(face_bioguide = bioguide,
           face_age = age,
           face_medianIncomeE = medianIncomeE,
           face_dpres = dpres,
           face_net_trump_vote = net_trump_vote,
           face_followers = followers,
           face_nokken_poole_dim1 = nokken_poole_dim1,
           face_nokken_poole_dim2 = nokken_poole_dim2,
           face_margin_pc = margin_pc,
           face_last_name = last_name,
           face_first_name = first_name,
           face_type = type,
           face_gender = gender,
           face_state = state,
           face_district = district,
           face_party = party,
           face_birthday = birthday)

data_merged <- left_join(data_merged, face_bio_covs)

# Create positive reaction variable
data_merged$pos_count <- data_merged$like_count + data_merged$love_count

# Create all reactions variable
data_merged$reactions <- data_merged$angry_count + data_merged$care_count + 
    data_merged$haha_count + data_merged$like_count + data_merged$love_count + 
    data_merged$sad_count + data_merged$share_count + data_merged$thankful_count + 
    data_merged$wow_count

# Create user face match variable (user bioguide matches face bioguide)
data_merged <- data_merged %>%
  mutate(user_face_match = face_bioguide == user_bioguide)

# Count and remove NA bioguides
na_bioguide_count <- sum(is.na(data_merged$user_bioguide))
if(na_bioguide_count > 0) {
    cat("Removing", na_bioguide_count, "observations with NA bioguides\n")
    data_merged <- data_merged[!is.na(data_merged$user_bioguide), ]
}

# Impute missing gender values
gender_imputations <- data.frame(
  user_bioguide = c("P000610", "I000056", "G000582", "E000285", "S000250", "V000129", "N000147"),
  user_gender = c("F", "M", "F", "M", "M", "M", "F")
)

# Update gender values in data_merged
for (i in 1:nrow(gender_imputations)) {
  data_merged$user_gender[data_merged$user_bioguide == gender_imputations$user_bioguide[i]] <- 
    gender_imputations$user_gender[i]
}

# Calculate number of observations per bioguide
obs_per_bioguide <- data_merged %>%
  group_by(face_bioguide) %>%
  summarise(n = n())

# Calculate the average number of observations per bioguide
avg_obs_per_bioguide <- mean(obs_per_bioguide$n)
print(avg_obs_per_bioguide)

# save to disk
save(data_merged, file = 'data/face_roberta.Rmd')



# Load data and functions -----------------------------------------------------------

# load data
load(file = 'data/face_roberta.Rmd')

# Define functions

# Define within fixed effects regression model function
fe_feols <- function(outcome_var, data, fixed_effects = NULL, cluster_var = NULL) {
    formula_string <- paste(outcome_var, "~", paste(independent_vars, collapse = " + "))
    
    # Add fixed effects if provided
    if (!is.null(fixed_effects)) {
        formula_string <- paste(formula_string, "|", paste(fixed_effects, collapse = " + "))
    }
    
    # Estimate the model
    model <- feols(as.formula(formula_string), data = data, cluster = cluster_var)
    
    return(model)
}

# Define within fixed effects model with HC1
fe_feols_hetero <- function(outcome_var, data, fixed_effects = NULL) {
    formula_string <- paste(outcome_var, "~", paste(independent_vars, collapse = " + "))
    
    # Add fixed effects if provided
    if (!is.null(fixed_effects)) {
        formula_string <- paste(formula_string, "|", paste(fixed_effects, collapse = " + "))
    }
    
    # Estimate the model
    model <- feols(as.formula(formula_string), data = data, vcov = "hetero")
    
    return(model)
}

# Updated lmr function with CR1 and nobs attribute
lmr <- function(outcome_var, data) {
    # Construct the formula string
    formula <- as.formula(paste(
      outcome_var, "~", 
      paste(independent_vars, collapse = " + "), 
      "+ (1 | user_bioguide)"
    ))
    
    # Fit the lmer model
    model <- lmer(formula, data = data)
    
    # Store number of observations
    nobs_val <- nobs(model)
    
    # Get the CR1 clustered variance-covariance matrix: use the cluster variable from model's frame
    clusters <- as.factor(model@frame$user_bioguide)
    vcov_cr1 <- vcovCR(model, type = "CR1", cluster = clusters)
    
    # Extract fixed effects and calculate CR1-based SEs, z-values and p-values
    coefs <- fixef(model)
    se <- sqrt(diag(vcov_cr1))
    zvals <- coefs / se
    pvals <- 2 * (1 - pnorm(abs(zvals)))
    
    # Create a result matrix
    result <- cbind(Estimate = coefs, "Std. Error" = se,
                    "z value" = zvals, "Pr(>|z|)" = pvals)
    
    # Attach attributes for later use
    attr(result, "nobs") <- nobs_val
    attr(result, "base_model") <- model
    
    # Assign a custom class so we can define methods like nobs
    class(result) <- c("cr1_model", class(result))
    
    return(result)
}

# Define S3 method for nobs on our new cr1_model class
nobs.cr1_model <- function(object, ...) {
  attr(object, "nobs")
}

# Function to filter terms for coeftest objects with CR1 SEs
term_filter <- function(model_obj, term_list, model_label) {
    if (is.matrix(model_obj) && all(c("Estimate", "Std. Error") %in% colnames(model_obj))) {
        coefs <- model_obj[,"Estimate"]
        ses <- model_obj[,"Std. Error"]
    } else if (inherits(model_obj, "coeftest")) {
        coefs <- coef(model_obj)
        ses <- model_obj[, "Std. Error"]
    } else {
        stop("Unsupported model type - expecting a matrix or coeftest object with 'Estimate' and 'Std. Error'")
    }

    model_df <- data.frame(
        term = names(coefs),
        estimate = coefs,
        std.error = ses,
        statistic = coefs / ses,
        model = model_label,
        stringsAsFactors = FALSE
    ) %>%
        filter(term %in% term_list)
    
    return(model_df)
}

# Function for stargazer to use CR1 SEs from coeftest objects
get_clustered_se_stargazer <- function(model) {
  if (is.matrix(model) && all(c("Estimate", "Std. Error") %in% colnames(model))) {
    return(model[, "Std. Error"])
  } else if (inherits(model, "coeftest")) {
    return(model[, "Std. Error"])
  } else {
    stop("Unsupported model type - expecting a numeric matrix or coeftest object with 'Estimate' and 'Std. Error'")
  }
}

# Define plot function
my_dwplot <- function(data, model_order, relabel_predictors, ggtitle, name, breaks, labels) {
  
  p <- dwplot(data,
              vline = geom_vline(xintercept = 0,
                                 colour = "red",
                                 linetype = 2),
              model_order = model_order,
              dot_args = list(size = 3),
              whisker_args = list(size = 1)) %>% 
    relabel_predictors(relabel_predictors) +
    theme_bw(base_size = 15) +
    xlab("Coefficient Estimate") + 
    ylab("") +
    ggtitle(ggtitle) +
    theme(
      plot.title = element_text(face = "bold"),
      #legend.position = c(-0.17, -0.2),
      legend.justification = c(0, 0),
      legend.background = element_rect(colour = "grey80"),
      legend.title.align = .5,
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank()) +
    scale_colour_grey(
      name = name,
      breaks = breaks,
      labels = labels
    )
  
  return(p)
}

# Define plot function for multiple models
my_comb_dwplot <- function(data, model_order, relabel_predictors, ggtitle, name, breaks, labels, y_axis_label) {
    
    p <- dwplot(data,
                vline = geom_vline(xintercept = 0,
                                   colour = "red",
                                   linetype = 2),
                model_order = model_order,
                dot_args = list(size = 3),
                whisker_args = list(size = 1)) %>% 
        relabel_predictors(relabel_predictors) +
        theme_bw(base_size = 15) +
        xlab("Coefficient Estimate") + 
        ylab("") +
        ggtitle(ggtitle) +
        theme(
            plot.title = element_text(face = "bold"),
            #legend.position = c(-0.17, -0.2),
            legend.justification = c(0, 0),
            legend.background = element_rect(colour = "grey80"),
            legend.title.align = .5,
            panel.grid.major = element_blank(), 
            panel.grid.minor = element_blank()) +
        scale_colour_grey(
            name = name,
            breaks = breaks,
            labels = labels)
    
    # Add custom y-axis label
    p <- p + ylab(y_axis_label)
    
    return(p)
}

# Function to remove prefixes and create a unified variable name
unify_variable_names <- function(name) {
  gsub("user_|face_", "", name)
}

# Extract and modify model summaries
modify_model_summary <- function(model) {
  # Ensure model is a numeric matrix with required column names
  if (is.matrix(model) && all(c("Estimate", "Std. Error") %in% colnames(model))) {
    summary_data <- model
    rownames(summary_data) <- sapply(rownames(summary_data), unify_variable_names)
    return(summary_data)
  } else {
    stop("Unsupported model type for modify_model_summary; expecting a numeric matrix with 'Estimate' and 'Std. Error'")
  }
}

# Define negative binomial regression model function including fixed effects
nb_model_fe <- function(outcome_var, data, fixed_effects_var, cluster_var) {
  # Create the formula string with fixed effects
  formula_string <- paste(outcome_var, "~", paste(independent_vars, collapse = " + "), "+ factor(", fixed_effects_var, ")")
  
  # Fit the model
  model <- glm.nb(as.formula(formula_string), data = data, control = glm.control(maxit = 1000))
  
  return(model)
}

# Define function to calculate clustered standard errors
get_clustered_se <- function(model, data, cluster_var) {
    vcov_cr1 <- vcovCR(model, type = "CR1", cluster = data[[cluster_var]])
    return(sqrt(diag(vcov_cr1)))
}




# Descriptive statistics --------------------------------------------------
# Load Descriptive Data --------------------------------

# Generate relevant sample
desc_sample <- data_merged %>% 
  filter(user_party == "Democratic" | user_party == "Republican") %>% 
  filter(user_face_match == TRUE) %>% 
  arrange(user_bioguide, date)

# Get list of user_bioguides
bio_list <- desc_sample %>%
  distinct(user_bioguide) %>%
  pull(user_bioguide)

# Bring in entire corpus from facebook_sentiment_roberta.csv and filter by bioguides in bio_list. Create new df
fb_corpus <- read_csv("output/facebook_sentiment_roberta.csv")
filtered_corpus <- fb_corpus %>%
  filter(bioguide %in% bio_list)
corpus_df <- filtered_corpus
names(corpus_df)

# Density plots (Figure 1) --------------------------------------------------

# A) Summarize the total number of posts per bioguide for each gender
post_counts <- fb_corpus %>%
  group_by(bioguide, gender) %>%
  summarise(total_posts = n()) %>%
  ungroup()

# Create a density plot
plot_post_counts <- ggplot(post_counts, aes(x = total_posts, fill = gender)) +
  geom_density(alpha = 0.75) +
  scale_fill_manual(values = c("M" = "skyblue", "F" = "orange"),
                    labels = c("M" = "Men", "F" = "Women")) +
  labs(title = "Total Posts",
       x = "Total Posts",
       y = "Density") +
  theme_minimal() +  # For a minimal theme with a white background
  theme(legend.position = "bottom",
        legend.title = element_blank())

# B) Summarize the total number of posts with images per bioguide for each gender
image_post_counts <- fb_corpus %>%
  filter(image_flag == 1) %>%
  group_by(bioguide, gender) %>%
  summarise(total_image_posts = n()) %>%
  ungroup()

# Create a density plot for posts with images
plot_image_post_counts <- ggplot(image_post_counts, aes(x = total_image_posts, fill = gender)) +
  geom_density(alpha = 0.75) +
  scale_fill_manual(values = c("M" = "skyblue", "F" = "orange"),
                    labels = c("M" = "Men", "F" = "Women")) +
  labs(title = "Total Image Posts",
       x = "Total Image Posts",
       y = "Density") +
  theme_minimal() +  # For a minimal theme with a white background
  theme(legend.position = "bottom",
        legend.title = element_blank())

# Calculate total reactions for each post
temp_corpus <- fb_corpus %>%
  mutate(total_reactions = angry_count + care_count + haha_count + 
           like_count + love_count + sad_count + wow_count + 
           thankful_count + share_count)

# Summarize the total reactions per bioguide for each gender
reactions_counts <- temp_corpus %>%
  group_by(bioguide, gender) %>%
  summarise(total_reactions = sum(total_reactions)) %>%
  ungroup()

# Create a density plot for total reactions
plot_reaction_counts <- ggplot(reactions_counts, aes(x = log(total_reactions), fill = gender)) +
  geom_density(alpha = 0.75) +
  scale_fill_manual(values = c("M" = "skyblue", "F" = "orange"),
                    labels = c("M" = "Men", "F" = "Women")) +
  labs(title = "Total Reactions (ln)",
       x = "Total Reactions",
       y = "Density") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

# Summarize the average face happiness per bioguide for each gender
average_happiness <- desc_sample %>%
  group_by(user_bioguide, user_gender) %>%
  summarise(average_happiness = mean(happiness, na.rm = TRUE)) %>%
  ungroup()

# Create a density plot for average face happiness
plot_avg_happiness <- ggplot(average_happiness, aes(x = average_happiness, fill = user_gender)) +
  geom_density(alpha = 0.75) +
  scale_fill_manual(values = c("M" = "skyblue", "F" = "orange"),
                    labels = c("M" = "Men", "F" = "Women")) +
  labs(title = "Average Face Happiness",
       x = "Average Face Happiness",
       y = "Density") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

# Summarize the average text positivity per bioguide for each gender
average_text_positivity <- fb_corpus %>%
  group_by(bioguide, gender) %>%
  summarise(average_positivity = mean(roberta_positive, na.rm = TRUE)) %>%
  ungroup()

# Create a density plot for average text positivity
plot_avg_text_positivity <- ggplot(average_text_positivity, aes(x = average_positivity, fill = gender)) +
  geom_density(alpha = 0.75) +
  scale_fill_manual(values = c("M" = "skyblue", "F" = "orange"),
                    labels = c("M" = "Men", "F" = "Women")) +
  labs(title = "Average Text Positivity",
       x = "Average Text Positivity",
       y = "Density") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

plot_avg_text_positivity

# Combine the plots with a specific layout
combined_plot <- (plot_avg_happiness | plot_avg_text_positivity) /
  (plot_post_counts | plot_image_post_counts | plot_reaction_counts)

# Arrange the layout and use guides only from the last plot
combined_plot <- combined_plot + plot_layout(guides = 'collect') & 
  theme(legend.position = "bottom") &
  guides(fill = guide_legend(title = NULL))

# Save the combined plot to a file
ggsave("output/plots/descriptive_plots.png", combined_plot, width = 10, height = 8)




# Box Plots (Appendix 3) ----------------------------------------------
# Box plot for Total Posts
boxplot_post_counts <- ggplot(post_counts, aes(x = gender, y = total_posts, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c("F" = "orange", "M" = "skyblue"),
                    labels = c("M" = "Men", "F" = "Women")) +
  scale_x_discrete(breaks = c("F", "M"), labels = c("Women", "Men")) +
  labs(title = "",
       x = "",
       y = "Total Posts") +
  theme_minimal()

boxplot_post_counts

# Box plot for Total Image Posts
boxplot_image_post_counts <- ggplot(image_post_counts, aes(x = gender, y = total_image_posts, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c("F" = "orange", "M" = "skyblue"),
                    labels = c("M" = "Men", "F" = "Women")) +
  scale_x_discrete(breaks = c("F", "M"), labels = c("Women", "Men")) +
  labs(title = "",
       x = "",
       y = "Total Image Posts") +
  theme_minimal()

# Box plot for Total Reactions (log-transformed for better visualization)
boxplot_reaction_counts <- ggplot(reactions_counts, aes(x = gender, y = log(total_reactions), fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c("F" = "orange", "M" = "skyblue"),
                    labels = c("M" = "Men", "F" = "Women")) +
  scale_x_discrete(breaks = c("F", "M"), labels = c("Women", "Men")) +
  labs(title = "",
       x = "",
       y = "Log of Total Reactions") +
  theme_minimal()

# Box plot for Average Face Happiness
boxplot_avg_happiness <- ggplot(average_happiness, aes(x = user_gender, y = average_happiness, fill = user_gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c("F" = "orange", "M" = "skyblue"),
                    labels = c("M" = "Men", "F" = "Women")) +
  scale_x_discrete(breaks = c("F", "M"), labels = c("Women", "Men")) +
  labs(title = "",
       x = "",
       y = "Average Face Happiness") +
  theme_minimal()

# Box plot for Average Text Positivity
boxplot_avg_text_positivity <- ggplot(average_text_positivity, aes(x = gender, y = average_positivity, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c("F" = "orange", "M" = "skyblue"),
                    labels = c("M" = "Men", "F" = "Women")) +
  scale_x_discrete(breaks = c("F", "M"), labels = c("Women", "Men")) +
  labs(title = "",
       x = "",
       y = "Average Text Positivity") +
  theme_minimal()

# Save box plots to disk
combined_boxplots <- boxplot_avg_happiness | boxplot_avg_text_positivity | boxplot_post_counts | boxplot_image_post_counts | boxplot_reaction_counts
combined_boxplots <- combined_boxplots + plot_layout(guides = 'collect') &
  theme(legend.position = "none") 

# Save the combined plot to a file
ggsave("output/plots/descriptive_boxplots.png", combined_boxplots, width = 10, height = 5)







# Gender differences ------------------------------------------------------
# Results 2a Women faces are happier than men (selfies)? -----------------------------------------------------------------

independent_vars <- c("relevel(as.factor(user_gender), ref = 'M')", "user_type", 
                      "user_party", "scale(user_age)", "scale(user_medianIncomeE)", "scale(user_dpres)", 
                      "scale(user_nokken_poole_dim1)", "scale(user_nokken_poole_dim2)", "scale(user_margin_pc)", 
                      "scale(user_followers)", "scale(user_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_allmocs <- data_merged %>% 
  filter(user_party == "Democratic" | user_party == "Republican") %>% 
  filter(user_face_match == TRUE) %>% 
    arrange(user_bioguide, date)

m1 <- lmr("scale(happiness)", fullmodel_allmocs)

term_list <- c('relevel(as.factor(user_gender), ref = "M")F')

m1_terms <- term_filter(m1, term_list, "Happiness")

plot1 <- my_dwplot(data = m1_terms,
                  model_order = c("Happiness"),
                  relabel_predictors = c('relevel(as.factor(user_gender), ref = "M")F' = "Woman"),
                  ggtitle = "Gender differences in emotions in self-presentation",
                  name = "Emotion",
                  breaks = c("Happiness"),
                  labels = c("Happiness"))
plot1

# Calculate substantive effect of Woman coefficient
sink("output/gender_effects_interpretation.txt")

cat("Interpreting Gender Effect on Happiness\n")
cat("====================================\n\n")

# Get original SD of happiness before scaling
orig_sd <- sd(fullmodel_allmocs$happiness, na.rm = TRUE)

# Get coefficient and SE by directly index the output matrix from lmr
woman_coef <- m1["relevel(as.factor(user_gender), ref = \"M\")F", "Estimate"]
woman_se   <- m1["relevel(as.factor(user_gender), ref = \"M\")F", "Std. Error"]

# Calculate effect in original units
effect_orig <- woman_coef * orig_sd

cat("Effect in standard deviations:\n")
cat(sprintf("%.3f SD (SE: %.3f)\n", woman_coef, woman_se))
cat(sprintf("95%% CI: [%.3f, %.3f]\n\n", 
            woman_coef - 1.96*woman_se, 
            woman_coef + 1.96*woman_se))

cat("Effect in original happiness units (0-1 scale):\n")
cat(sprintf("%.3f units\n", effect_orig))
cat(sprintf("This means women display %.1f%% more happiness than men\n", 
            100 * effect_orig))

sink()

# Optional: Add a message to confirm where output was saved
cat(sprintf("Results have been saved to %s\n", "output/gender_effects_interpretation.txt"))

# Results 2b Women faces are happier than men (all faces)? -----------------------------------------------------------------

independent_vars <- c("relevel(as.factor(face_gender), ref = 'M')", "face_type", 
                      "face_party", "scale(face_age)", "scale(face_medianIncomeE)", "scale(face_dpres)", 
                      "scale(face_nokken_poole_dim1)", "scale(face_nokken_poole_dim2)", "scale(face_margin_pc)", 
                      "scale(face_followers)", "scale(face_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_allmocs <- data_merged %>% 
    filter(face_party == "Democratic" | face_party == "Republican") %>% 
    arrange(face_bioguide, date)

m2 <- lmr("scale(happiness)", fullmodel_allmocs)

term_list <- c('relevel(as.factor(face_gender), ref = "M")F')

m2_terms <- term_filter(m2, term_list, "Happiness")

plot2 <- my_dwplot(data = m2_terms,
                  model_order = c("Happiness"),
                  relabel_predictors = c('relevel(as.factor(face_gender), ref = "M")F' = "Woman"),
                  ggtitle = "Gender differences in emotions for all MOC faces",
                  name = "Emotion",
                  breaks = c("Happiness"),
                  labels = c("Happiness"))
plot2


# Results 2c: Women write more positive text? ------------------------------

independent_vars <- c("scale(happiness)", "relevel(as.factor(user_gender), ref = 'M')", "user_type", 
                      "user_party", "scale(user_age)", "scale(user_medianIncomeE)", "scale(user_dpres)", 
                      "scale(user_nokken_poole_dim1)", "scale(user_nokken_poole_dim2)", "scale(user_margin_pc)", 
                      "scale(user_followers)", "scale(user_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_allmocs <- data_merged %>% 
    filter(user_party == "Democratic" | user_party == "Republican") %>% 
    filter(user_face_match == TRUE) %>% 
    arrange(user_bioguide, date)

m3 <- lmr("scale(roberta_positive)", fullmodel_allmocs)

term_list <- c('relevel(as.factor(user_gender), ref = "M")F')

m3_terms <- term_filter(m3, term_list, "Happiness")

plot3 <- my_dwplot(data = m3_terms,
                   model_order = c("Happiness"),
                   relabel_predictors = c('relevel(as.factor(user_gender), ref = "M")F' = "Woman"),
                   ggtitle = "Gender differences in positive text",
                   name = "Emotion",
                   breaks = c("Happiness"),
                   labels = c("Happiness"))

plot3 <- plot3 + theme(legend.position = "none") + scale_x_continuous(limits = c(-0.1, 0.5))

# Combine Results 2 plots (Figure 2) -------------------------------------------------


combined_terms <- rbind(m1_terms, m2_terms)
combined_terms$model <- rep(c("Woman: Selfie", "Woman: All"), each = nrow(m1_terms))

combined_terms <- combined_terms %>%
    mutate(term = if_else(model == "Woman: Selfie", "Woman: Selfie", "Woman: All"))

combined_plot <- my_comb_dwplot(data = combined_terms,
                                model_order = c("Woman: Selfie", "Woman: All"),
                                relabel_predictors = c("Woman: Selfie" = "Woman: Selfie", "Woman: All" = "Woman: All"),
                                ggtitle = "Gender differences in happiness in images",
                                name = "Emotion",
                                breaks = c("Woman: Selfie", "Woman: All"),
                                labels = c("Woman: Selfie", "Woman: All"),
                                y_axis_label = "")

combined_plot <- combined_plot + theme(legend.position = "none")

fcombined_plot <- plot_grid(combined_plot, plot3, 
                            nrow = 2, 
                            align = "v", 
                            rel_heights = c(1, 1), 
                            labels = c("A", "B"))

print(fcombined_plot)
ggsave("output/plots/results2_combined_coef.pdf", fcombined_plot, width = 20, height = 14, units = "cm")


 
# Table of results for gender differences in facial happiness Appendices 5-7 (selfies, all faces, and text) -------------------------

# Apply the function to each model
m1_modified <- modify_model_summary(m1)
m2_modified <- modify_model_summary(m2)
m3_modified <- modify_model_summary(m3)

# Export individual tables with CR1 clustered SEs
stargazer(m1_modified, 
          out = "output/tables/table_a1_selfies.tex",
          title = "Gender differences in facial displays of happiness",
          se = list(get_clustered_se_stargazer(m1)))

stargazer(m2_modified, 
          out = "output/tables/table_a2_all_faces.tex",
          title = "Gender differences in facial displays of happiness (all faces)",
          se = list(get_clustered_se_stargazer(m2)))

stargazer(m3_modified, 
          out = "output/tables/table_a3_text_positivity.tex",
          title = "Gender differences in text positivity",
          se = list(get_clustered_se_stargazer(m3)))

# Combined table with CR1 clustered SEs
stargazer(m1_modified, m2_modified, m3_modified,
          out = "output/tables/table_combined.tex",
          title = "Combined Results",
          se = list(get_clustered_se_stargazer(m1),
                   get_clustered_se_stargazer(m2),
                   get_clustered_se_stargazer(m3)))




# Audience models ---------------------------------------------------------
# Results 4a Audience responds to happiness (Women) --------------------------------

independent_vars <- c("scale(happiness)", "scale(roberta_positive)", 
                      "scale(roberta_negative)")

fullmodel_allmocs <- data_merged %>% 
    filter(user_party == "Democratic" | user_party == "Republican") %>% 
    filter(user_face_match == TRUE) %>% 
    filter(user_gender == "F") %>% 
    arrange(user_bioguide, date)

# Fit models with fixed effects for user_bioguide
aud_w_m1 <- nb_model_fe("pos_count", fullmodel_allmocs, "user_bioguide", "user_bioguide")
aud_w_m2 <- nb_model_fe("like_count", fullmodel_allmocs, "user_bioguide", "user_bioguide")
aud_w_m3 <- nb_model_fe("reactions", fullmodel_allmocs, "user_bioguide", "user_bioguide")

# Calculate clustered standard errors
se_aud_w_m1 <- get_clustered_se(aud_w_m1, fullmodel_allmocs, "user_bioguide")
se_aud_w_m2 <- get_clustered_se(aud_w_m2, fullmodel_allmocs, "user_bioguide")
se_aud_w_m3 <- get_clustered_se(aud_w_m3, fullmodel_allmocs, "user_bioguide")

# Calculate clustered standard errors for audience models
ct_aud_w_m1 <- coeftest(aud_w_m1, vcov. = vcovCR(aud_w_m1, type = "CR1", 
                        cluster = fullmodel_allmocs$user_bioguide))
ct_aud_w_m2 <- coeftest(aud_w_m2, vcov. = vcovCR(aud_w_m2, type = "CR1", 
                        cluster = fullmodel_allmocs$user_bioguide))
ct_aud_w_m3 <- coeftest(aud_w_m3, vcov. = vcovCR(aud_w_m3, type = "CR1", 
                        cluster = fullmodel_allmocs$user_bioguide))

term_list <- c("scale(happiness)", "scale(roberta_positive)")

aud_w_m1_terms <- term_filter(ct_aud_w_m1, term_list, "Positive")
aud_w_m2_terms <- term_filter(ct_aud_w_m2, term_list, "Likes")
aud_w_m3_terms <- term_filter(ct_aud_w_m3, term_list, "Reactions")
combined_terms <- rbind(aud_w_m1_terms, aud_w_m2_terms, aud_w_m3_terms)

plot4f <- my_dwplot(data = combined_terms,
                   model_order = c("Positive", "Likes", "Reactions"),
                   relabel_predictors = c("scale(happiness)" = "Face: Happiness",
                                          "scale(roberta_positive)" = "Text: Positive"),
                   ggtitle = "Social media reactions to emotion, Women MOCs",
                   name = "Reaction",
                   breaks = c("Reactions", "Likes", "Positive"),
                   labels = c("Any reaction", "Likes", "Positive"))

plot4f <- plot4f + theme(legend.position = "none") + xlim(-0.1, 0.5)
plot4f

# Results 4b Audience responds to happiness (Men) --------------------------------

independent_vars <- c("scale(happiness)", "scale(roberta_positive)", 
                      "scale(roberta_negative)")

fullmodel_allmocs <- data_merged %>% 
  filter(user_party == "Democratic" | user_party == "Republican") %>% 
  filter(user_face_match == TRUE) %>% 
  filter(user_gender == "M") %>% 
  arrange(user_bioguide, date)

# Fit models with fixed effects for user_bioguide
aud_m_m1 <- nb_model_fe("pos_count", fullmodel_allmocs, "user_bioguide", "user_bioguide")
aud_m_m2 <- nb_model_fe("like_count", fullmodel_allmocs, "user_bioguide", "user_bioguide")
aud_m_m3 <- nb_model_fe("reactions", fullmodel_allmocs, "user_bioguide", "user_bioguide")

# Calculate clustered standard errors
se_aud_m_m1 <- get_clustered_se(aud_m_m1, fullmodel_allmocs, "user_bioguide")
se_aud_m_m2 <- get_clustered_se(aud_m_m2, fullmodel_allmocs, "user_bioguide")
se_aud_m_m3 <- get_clustered_se(aud_m_m3, fullmodel_allmocs, "user_bioguide")

# Calculate clustered standard errors for audience models
ct_aud_m_m1 <- coeftest(aud_m_m1, vcov. = vcovCR(aud_m_m1, type = "CR1", 
                        cluster = fullmodel_allmocs$user_bioguide))
ct_aud_m_m2 <- coeftest(aud_m_m2, vcov. = vcovCR(aud_m_m2, type = "CR1", 
                        cluster = fullmodel_allmocs$user_bioguide))
ct_aud_m_m3 <- coeftest(aud_m_m3, vcov. = vcovCR(aud_m_m3, type = "CR1", 
                        cluster = fullmodel_allmocs$user_bioguide))

term_list <- c("scale(happiness)", "scale(roberta_positive)")

aud_m_m1_terms <- term_filter(ct_aud_m_m1, term_list, "Positive")
aud_m_m2_terms <- term_filter(ct_aud_m_m2, term_list, "Likes")
aud_m_m3_terms <- term_filter(ct_aud_m_m3, term_list, "Reactions")
combined_terms <- rbind(aud_m_m1_terms, aud_m_m2_terms, aud_m_m3_terms)

plot4m <- my_dwplot(data = combined_terms,
                    model_order = c("Positive", "Likes", "Reactions"),
                    relabel_predictors = c("scale(happiness)" = "Face: Happiness",
                                           "scale(roberta_positive)" = "Text: Positive"),
                    ggtitle = "Social media reactions to emotion, Men MOCs",
                    name = "Reaction",
                    breaks = c("Reactions", "Likes", "Positive"),
                    labels = c("Any reaction", "Likes", "Positive"))

plot4m <- plot4m + theme(legend.position = "bottom") + xlim(-0.1, 0.5)
plot4m

# Results4 combined plot (Figure 3) --------------------------------------------------

fcombined_plot <- plot_grid(plot4f, plot4m, 
                            nrow = 2, 
                            align = "v", 
                            rel_heights = c(1, 1.2), 
                            labels = c("A", "B"))

print(fcombined_plot)
ggsave("output/plots/results4_combined_coef.pdf", fcombined_plot, width = 20, height = 18, units = "cm")



# Full results ------------------------------------------------------------

# A.5 Facial displays of happiness, MOC-posted images only ----------------
independent_vars <- c("relevel(as.factor(user_gender), ref = 'M')", "user_type", 
                      "user_party", "scale(user_age)", "scale(user_medianIncomeE)", "scale(user_dpres)", 
                      "scale(user_nokken_poole_dim1)", "scale(user_nokken_poole_dim2)", "scale(user_margin_pc)", 
                      "scale(user_followers)", "scale(user_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_allmocs <- data_merged %>% 
  filter(user_party == "Democratic" | user_party == "Republican") %>% 
  filter(user_face_match == TRUE) %>% 
  arrange(user_bioguide, date)

m1 <- lmr("scale(happiness)", fullmodel_allmocs)

term_list <- c('relevel(as.factor(user_gender), ref = "M")F', 'user_typeSEN', 'user_partyRepublican', 'scale(user_age)',
               'scale(user_medianIncomeE)', 'scale(user_dpres)', 'scale(user_nokken_poole_dim1)', 'scale(user_nokken_poole_dim2)',
               'scale(user_margin_pc)', 'scale(user_followers)', 'scale(user_net_trump_vote)', 'face_race_aa', 'face_race_la')

m1_terms <- term_filter(m1, term_list, "Happiness")

plot1 <- my_dwplot(data = m1_terms,
                   model_order = c("Happiness"),
                   relabel_predictors = c('relevel(as.factor(user_gender), ref = "M")F' = "Woman",
                                          'user_typeSEN' = "Senator",
                                          'user_partyRepublican' = "Republican",
                                          'scale(user_age)' = "Age",
                                          'scale(user_medianIncomeE)' = "Median Income",
                                          'scale(user_dpres)' = "Clinton Vote %",
                                          'scale(user_nokken_poole_dim1)' = "Ideology Dim-1",
                                          'scale(user_nokken_poole_dim2)' = "Ideology Dim-2",
                                          'scale(user_margin_pc)' = "Vote Margin",
                                          'scale(user_followers)' = "Followers",
                                          'scale(user_net_trump_vote)' = "Trump Fidelity",
                                          'face_race_aa' = "Afr. American",
                                          'face_race_la' = "Latino"),
                   ggtitle = "Gender differences in facial happiness\nin self-presentation",
                   name = "Emotion",
                   breaks = c("Happiness"),
                   labels = c("Happiness"))

plot1 <- plot1 + theme(legend.position = "none")
ggsave("output/plots/results2a_coef_full.pdf", width = 20, height = 14, units = "cm")

# A.6 Facial displays of happiness in all MOC images ----------------------

independent_vars <- c("relevel(as.factor(face_gender), ref = 'M')", "face_type", 
                      "face_party", "scale(face_age)", "scale(face_medianIncomeE)", "scale(face_dpres)", 
                      "scale(face_nokken_poole_dim1)", "scale(face_nokken_poole_dim2)", "scale(face_margin_pc)", 
                      "scale(face_followers)", "scale(face_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_allmocs <- data_merged %>% 
  filter(face_party == "Democratic" | face_party == "Republican") %>% 
  arrange(face_bioguide, date)

m2 <- lmr("scale(happiness)", fullmodel_allmocs)

term_list <- c('relevel(as.factor(face_gender), ref = "M")F', 'face_typeSEN', 'face_partyRepublican', 'scale(face_age)',
               'scale(face_medianIncomeE)', 'scale(face_dpres)', 'scale(face_nokken_poole_dim1)', 'scale(face_nokken_poole_dim2)',
               'scale(face_margin_pc)', 'scale(face_followers)', 'scale(face_net_trump_vote)', 'face_race_aa', 'face_race_la')

m2_terms <- term_filter(m2, term_list, "Happiness")

plot2 <- my_dwplot(data = m2_terms,
                   model_order = c("Happiness"),
                   relabel_predictors = c('relevel(as.factor(face_gender), ref = "M")F' = "Woman",
                                          'face_typeSEN' = "Senator",
                                          'face_partyRepublican' = "Republican",
                                          'scale(face_age)' = "Age",
                                          'scale(face_medianIncomeE)' = "Median Income",
                                          'scale(face_dpres)' = "Clinton Vote %",
                                          'scale(face_nokken_poole_dim1)' = "Ideology Dim-1",
                                          'scale(face_nokken_poole_dim2)' = "Ideology Dim-2",
                                          'scale(face_margin_pc)' = "Vote Margin",
                                          'scale(face_followers)' = "Followers",
                                          'scale(face_net_trump_vote)' = "Trump Fidelity",
                                          'face_race_aa' = "Afr. American",
                                          'face_race_la' = "Latino"),
                   ggtitle = "Gender differences in facial displays of\nhappiness for all MOC faces",
                   name = "Emotion",
                   breaks = c("Happiness"),
                   labels = c("Happiness"))
plot2 <- plot2 + theme(legend.position = "none")
ggsave("output/plots/results2b_coef_full.pdf", width = 20, height = 14, units = "cm")




# A.7 Text positivity and MOC gender --------------------------------------



independent_vars <- c("relevel(as.factor(user_gender), ref = 'M')", "scale(happiness)", "user_type", 
                      "user_party", "scale(user_age)", "scale(user_medianIncomeE)", "scale(user_dpres)", 
                      "scale(user_nokken_poole_dim1)", "scale(user_nokken_poole_dim2)", "scale(user_margin_pc)", 
                      "scale(user_followers)", "scale(user_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_allmocs <- data_merged %>% 
  filter(user_party == "Democratic" | user_party == "Republican") %>% 
  filter(user_face_match == TRUE) %>% 
  arrange(user_bioguide, date)

m3 <- lmr("scale(roberta_positive)", fullmodel_allmocs)

term_list <- c('relevel(as.factor(user_gender), ref = "M")F', 'scale(happiness)', 'user_typeSEN', 'user_partyRepublican', 'scale(user_age)',
               'scale(user_medianIncomeE)', 'scale(user_dpres)', 'scale(user_nokken_poole_dim1)', 'scale(user_nokken_poole_dim2)',
               'scale(user_margin_pc)', 'scale(user_followers)', 'scale(user_net_trump_vote)', 'face_race_aa', 'face_race_la')

m3_terms <- term_filter(m3, term_list, "Happiness")

plot3 <- my_dwplot(data = m3_terms,
                   model_order = c("Happiness"),
                   relabel_predictors = c('relevel(as.factor(user_gender), ref = "M")F' = "Woman",
                                          'scale(happiness)' = "Face: Happiness",
                                          'user_typeSEN' = "Senator",
                                          'user_partyRepublican' = "Republican",
                                          'scale(user_age)' = "Age",
                                          'scale(user_medianIncomeE)' = "Median Income",
                                          'scale(user_dpres)' = "Clinton Vote %",
                                          'scale(user_nokken_poole_dim1)' = "Ideology Dim-1",
                                          'scale(user_nokken_poole_dim2)' = "Ideology Dim-2",
                                          'scale(user_margin_pc)' = "Vote Margin",
                                          'scale(user_followers)' = "Followers",
                                          'scale(user_net_trump_vote)' = "Trump Fidelity",
                                          'face_race_aa' = "Afr. American",
                                          'face_race_la' = "Latino"),
                   ggtitle = "Gender differences in positive text",
                   name = "Emotion",
                   breaks = c("Happiness"),
                   labels = c("Happiness"))

plot3 <- plot3 + theme(legend.position = "none")
ggsave("output/plots/results2c_coef_full.pdf", width = 20, height = 14, units = "cm")

# A.8(a) Democratic gender differences in facial displays of happiness for all MOC faces --------

independent_vars <- c("relevel(as.factor(face_gender), ref = 'M')", "face_type", 
                      "scale(face_age)", "scale(face_medianIncomeE)", "scale(face_dpres)", 
                      "scale(face_nokken_poole_dim1)", "scale(face_nokken_poole_dim2)", "scale(face_margin_pc)", 
                      "scale(face_followers)", "scale(face_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_republicans <- data_merged %>% 
  filter(face_party == "Democratic") %>% 
  arrange(face_bioguide, date)

m5e <- lmr("scale(happiness)", fullmodel_republicans)

term_list <- c('relevel(as.factor(face_gender), ref = "M")F', 'face_typeSEN', 'scale(face_age)',
               'scale(face_medianIncomeE)', 'scale(face_dpres)', 'scale(face_nokken_poole_dim1)', 'scale(face_nokken_poole_dim2)',
               'scale(face_margin_pc)', 'scale(face_followers)', 'scale(face_net_trump_vote)', 'face_race_aa', 'face_race_la')

m5e_terms <- term_filter(m5e, term_list, "Happiness")

plotm5e <- my_dwplot(data = m5e_terms,
                     model_order = c("Happiness"),
                     relabel_predictors = c('relevel(as.factor(face_gender), ref = "M")F' = "Woman",
                                            'face_typeSEN' = "Senator",
                                            'scale(face_age)' = "Age",
                                            'scale(face_medianIncomeE)' = "Median Income",
                                            'scale(face_dpres)' = "Clinton Vote %",
                                            'scale(face_nokken_poole_dim1)' = "Ideology Dim-1",
                                            'scale(face_nokken_poole_dim2)' = "Ideology Dim-2",
                                            'scale(face_margin_pc)' = "Vote Margin",
                                            'scale(face_followers)' = "Followers",
                                            'scale(face_net_trump_vote)' = "Trump Fidelity",
                                            'face_race_aa' = "Afr. American",
                                            'face_race_la' = "Latino"),
                     ggtitle = "Democratic gender differences in facial displays of\nhappiness for all MOC faces",
                     name = "Emotion",
                     breaks = c("Happiness"),
                     labels = c("Happiness"))

plotm5e <- plotm5e + theme(legend.position = "none")
ggsave("output/plots/append_party_dem.pdf", width = 20, height = 14, units = "cm")










# Table A4: Democratic gender differences
m5e_modified <- modify_model_summary(m5e)
stargazer(m5e_modified, 
          out = "output/tables/table_a4_democratic_differences.tex",
          title = "Democratic gender differences in facial displays of happiness",
          se = list(get_clustered_se_stargazer(m5e)))



# A.8(b) Republican gender differences in facial displays of happiness for all MOC faces --------

independent_vars <- c("relevel(as.factor(face_gender), ref = 'M')", "face_type", 
                      "scale(face_age)", "scale(face_medianIncomeE)", "scale(face_dpres)", 
                      "scale(face_nokken_poole_dim1)", "scale(face_nokken_poole_dim2)", "scale(face_margin_pc)", 
                      "scale(face_followers)", "scale(face_net_trump_vote)", "face_race_aa", "face_race_la")

fullmodel_republicans <- data_merged %>% 
  filter(face_party == "Republican") %>% 
  arrange(face_bioguide, date)

m5b <- lmr("scale(happiness)", fullmodel_republicans)

term_list <- c('relevel(as.factor(face_gender), ref = "M")F', 'face_typeSEN', 'scale(face_age)',
               'scale(face_medianIncomeE)', 'scale(face_dpres)', 'scale(face_nokken_poole_dim1)', 'scale(face_nokken_poole_dim2)',
               'scale(face_margin_pc)', 'scale(face_followers)', 'scale(face_net_trump_vote)', 'face_race_aa', 'face_race_la')

m5b_terms <- term_filter(m5b, term_list, "Happiness")

plot5b <- my_dwplot(data = m5b_terms,
                    model_order = c("Happiness"),
                    relabel_predictors = c('relevel(as.factor(face_gender), ref = "M")F' = "Woman",
                                           'face_typeSEN' = "Senator",
                                           'scale(face_age)' = "Age",
                                           'scale(face_medianIncomeE)' = "Median Income",
                                           'scale(face_dpres)' = "Clinton Vote %",
                                           'scale(face_nokken_poole_dim1)' = "Ideology Dim-1",
                                           'scale(face_nokken_poole_dim2)' = "Ideology Dim-2",
                                           'scale(face_margin_pc)' = "Vote Margin",
                                           'scale(face_followers)' = "Followers",
                                           'scale(face_net_trump_vote)' = "Trump Fidelity",
                                           'face_race_aa' = "Afr. American",
                                           'face_race_la' = "Latino"),
                    ggtitle = "Republican gender differences in facial displays of\nhappiness for all MOC faces",
                    name = "Emotion",
                    breaks = c("Happiness"),
                    labels = c("Happiness"))

plot5b <- plot5b + theme(legend.position = "none")
ggsave("output/plots/append_party_republican.pdf", width = 20, height = 14, units = "cm")



# Table A5: Republican gender differences
m5b_modified <- modify_model_summary(m5b)
stargazer(m5b_modified, 
          out = "output/tables/table_a5_republican_differences.tex",
          title = "Republican gender differences in facial displays of happiness",
          se = list(get_clustered_se_stargazer(m5b)))



# Table A6: Summary statistics of social media reactions
stargazer(reactions_data, 
          out = "output/tables/table_a6_reactions_summary.tex",
          type = "latex", 
          summary = TRUE, 
          summary.stat = c("min", "p25", "median", "mean", "p75", "max"))



# Table A7: Social media reactions to emotion, Women MOCs
modelsummary(list(ct_aud_w_m1, ct_aud_w_m2, ct_aud_w_m3),
            output = "output/tables/table_a7_women_reactions.tex",
            column_labels = c("Reactions", "Likes", "Positive"),
            stars = c("***" = 0.01, "**" = 0.05, "*" = 0.1),
            gof_omit = "AIC|BIC|RMSE",
            tidy = tidy_custom)

# Table A8: Social media reactions to emotion, Men MOCs
modelsummary(list(ct_aud_m_m1, ct_aud_m_m2, ct_aud_m_m3),
            output = "output/tables/table_a8_men_reactions.tex",
            column_labels = c("Reactions", "Likes", "Positive"),
            stars = c("***" = 0.01, "**" = 0.05, "*" = 0.1),
            gof_omit = "AIC|BIC|RMSE",
            tidy = tidy_custom)

# Create a report file for model observations
sink("output/model_observations_report.txt")

cat("Dataset Observations\n")
cat("===================\n")
cat("Initial imported data (data):", nrow(data), "\n")
cat("Merged dataset (data_merged):", nrow(data_merged), "\n")
cat("Facebook corpus (fb_corpus):", nrow(fb_corpus), "\n")
cat("Filtered corpus (filtered_corpus):", nrow(filtered_corpus), "\n\n")

# Detailed diagnostics for bioguide counts
all_bioguides <- unique(data_merged$user_bioguide)
women_bioguides <- unique(data_merged$user_bioguide[data_merged$user_gender == "F"])
men_bioguides <- unique(data_merged$user_bioguide[data_merged$user_gender == "M"])

cat("Unique Bioguides Count (with diagnostics)\n")
cat("======================================\n")
cat("Total unique bioguides:", length(all_bioguides), "\n")
cat("Unique women bioguides:", length(women_bioguides), "\n")
cat("Unique men bioguides:", length(men_bioguides), "\n")

# Check for any bioguides with NA or missing gender
na_gender_bioguides <- unique(data_merged$user_bioguide[is.na(data_merged$user_gender)])
if(length(na_gender_bioguides) > 0) {
    cat("\nBioguides with NA gender:", length(na_gender_bioguides), "\n")
    cat("They are:", paste(na_gender_bioguides, collapse=", "), "\n")
}

# Check for bioguides with multiple gender assignments
gender_check <- data_merged %>%
    group_by(user_bioguide) %>%
    summarize(
        genders = paste(sort(unique(user_gender)), collapse=", "),
        n_genders = n_distinct(user_gender)
    ) %>%
    filter(n_genders > 1)

if(nrow(gender_check) > 0) {
    cat("\nBioguides with multiple gender assignments:\n")
    print(gender_check)
}

# Verify the sum
cat("\nVerification:\n")
cat("Women + Men bioguides:", length(women_bioguides) + length(men_bioguides), "\n")
cat("Difference from total:", length(all_bioguides) - (length(women_bioguides) + length(men_bioguides)), "\n")

# List any bioguides that appear in neither women nor men lists
unaccounted <- setdiff(all_bioguides, c(women_bioguides, men_bioguides))
if(length(unaccounted) > 0) {
    cat("\nBioguides that are neither counted as men nor women:", length(unaccounted), "\n")
    cat("They are:", paste(unaccounted, collapse=", "), "\n")
}

cat("\n")

# Model Observation Counts
cat("Model Observation Counts\n")
cat("=====================\n")

# Results 2a - All MOC images
cat("Results 2a - All MOC images:\n")
cat("Total observations:", nobs(m2), "\n\n")

# Results 2a - Facial happiness in selfies
cat("Results 2a - Facial happiness in selfies:\n")
cat("Total observations:", nobs(m1), "\n\n")

# Results 3 - Text positivity and MOC gender
cat("Results 3 - Text positivity and MOC gender:\n")
cat("Total observations:", nobs(m3), "\n\n")

# Results 4 - Audience response models
cat("Results 4 - Audience response models:\n")
cat("Women MOCs observations:", nobs(aud_w_m1), "\n")
cat("Men MOCs observations:", nobs(aud_m_m1), "\n\n")

# Appendix Models
cat("Appendix Models:\n")
cat("Democratic gender differences (m5e):", nobs(m5e), "\n")
cat("Republican gender differences (m5b):", nobs(m5b), "\n\n")

# Add diagnostics for potential issues
multi_gender_bioguides <- data_merged %>%
  group_by(user_bioguide) %>%
  summarize(unique_genders = n_distinct(user_gender)) %>%
  filter(unique_genders > 1)

if(nrow(multi_gender_bioguides) > 0) {
  cat("\nBioguides with multiple gender assignments:\n")
  for(bg in multi_gender_bioguides$user_bioguide) {
    genders <- unique(data_merged$user_gender[data_merged$user_bioguide == bg])
    cat(sprintf("%s: %s\n", bg, paste(genders, collapse=", ")))
  }
}

sink() 



