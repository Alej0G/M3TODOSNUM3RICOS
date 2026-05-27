class MatrixError(Exception):
    """Error relacionado con propiedades de la matriz."""
    pass

class ConvergenceError(Exception):
    """El método no converge con los parámetros dados."""
    pass

class ValidationError(Exception):
    """Error en los datos de entrada del usuario."""
    pass


def format_error_response(errors: list[str]) -> dict:
    """
    Convierte una lista de mensajes de error en el formato
    estándar de respuesta JSON para el frontend.
    """
    return {
        "ok":     False,
        "errors": errors,
    }


def format_exception(e: Exception) -> dict:
    """Para excepciones inesperadas en los endpoints."""
    return {
        "ok":    False,
        "error": str(e),
    }