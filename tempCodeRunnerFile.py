
            with torch.no_grad():
                recon_out = model(window_tensor)
                reconstruction_error = torch.mean((window_tensor - recon_out) ** 2).item()

                if not pose_detected:
                    form_status = "No Pose Detected"
                    status_color = (0, 165, 255)
                elif reconstruction_error > ANOMALY_THRESHOLD:
                    viable_rep = False
                    form_status = "POOR FORM!"
                    status_color = (0, 0, 255)
                elif angle >= 100:
                    viable_rep = False
                    form_status = "Only lift to shoulder height!"
                    status_color = (0, 0, 255)
                else:
                    form_status = "Good Form"
                    status_color = (0, 255, 0)
        except Exception as e:
            print(f"Inference error: {e}")
            form_status = "Error"
            status_color = (0, 165, 255)