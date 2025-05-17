import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_video(video_path):
    """Process video to track the baseball and return positions"""
    # Read video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    ball_positions = []
    
    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find circular objects (potential baseballs)
        best_ball = None
        max_circularity = 0
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < 20 or area > 500:  # Adjust based on your video
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > max_circularity and circularity > 0.5:
                max_circularity = circularity
                best_ball = contour
        
        if best_ball is not None:
            # Get center of the ball
            M = cv2.moments(best_ball)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ball_positions.append((cx, cy, frame_count))
    
    cap.release()
    return np.array(ball_positions) if ball_positions else None

def extract_features(ball_positions):
    """Extract trajectory features from ball positions"""
    if ball_positions is None or len(ball_positions) < 5:
        return None
        
    # Extract x, y coordinates and frame numbers
    x = ball_positions[:, 0]
    y = ball_positions[:, 1]
    t = ball_positions[:, 2]
    
    # Calculate velocities (pixels per frame)
    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        velocities = np.sqrt(dx**2 + dy**2) / dt
    else:
        velocities = np.array([0])
    
    # Fit polynomial to trajectory (y as function of x)
    try:
        # 2nd degree polynomial fit for trajectory
        z = np.polyfit(x, y, 2)
        polynomial = np.poly1d(z)
        
        # Calculate fitted y values
        y_fit = polynomial(x)
        
        # Calculate curvature
        curvature = abs(z[0])
        
        # Calculate additional metrics
        total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        straight_line_distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        curve_ratio = total_distance / straight_line_distance if straight_line_distance > 0 else 1
        
        # Extract features
        features = {
            'vertical_drop': y[-1] - y[0],
            'horizontal_movement': x[-1] - x[0],
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'curvature': curvature,
            'curve_ratio': curve_ratio
        }
        
        return features
    
    except Exception as e:
        return None

def predict_pitch_type(features):
    """Predict pitch type based on extracted features"""
    if not features:
        return "unknown"
    
    # Extract key metrics
    curvature = features['curvature']
    horizontal_movement = abs(features['horizontal_movement'])
    vertical_drop = features['vertical_drop']
    curve_ratio = features['curve_ratio']
    
    # Simple rule-based classification
    # Note: These thresholds are based on the video's pixel space and would need calibration
    
    # Curveball - significant curvature and vertical drop
    if curvature > 0.005 and vertical_drop > 100:
        return "curveball"
    
    # Slider - moderate to high curvature with significant horizontal movement
    elif curvature > 0.002 and horizontal_movement > 80:
        return "slider"
    
    # Sinker - moderate downward and horizontal movement
    elif vertical_drop > 70 and horizontal_movement > 50 and curvature < 0.003:
        return "sinker"
    
    # Changeup - less velocity, moderate movement
    elif features['avg_velocity'] < 15 and curve_ratio < 1.1:
        return "changeup"
    
    # Fastball - high velocity, relatively straight
    else:
        return "fastball"

def analyze_pitch(video_path):
    """Analyze a pitch video and return the predicted pitch type"""
    # Track the ball
    ball_positions = process_video(video_path)
    
    if ball_positions is not None and len(ball_positions) > 0:
        # Extract features
        features = extract_features(ball_positions)
        
        if features:
            # Predict pitch type
            pitch_type = predict_pitch_type(features)
            return pitch_type
    
    return "unknown"

def main():
    """Main function to process a video and predict pitch type"""
    # Define the video path
    video_path = "/Users/ishan/Downloads/tanaka.mov"
    
    # Process the video and predict pitch type
    pitch_type = analyze_pitch(video_path)
    
    # Output prediction
    print(f"Predicted pitch type: {pitch_type}")

if __name__ == "__main__":
    main()