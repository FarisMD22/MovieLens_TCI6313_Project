"""
Main entry point for preprocessing
Run this file to preprocess the MovieLens 1M dataset
"""

from src.preprocessing import MovieLensPreprocessor


def main():
    """Main preprocessing function"""
    # Create preprocessor instance
    preprocessor = MovieLensPreprocessor()

    # Run full pipeline
    preprocessor.run_full_pipeline()

    print("\n✅ Preprocessing completed! You can now proceed to model training.")


if __name__ == "__main__":
    main()