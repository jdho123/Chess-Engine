import numpy as np
import orjson


class ChessFeatureExtractor:
    """Extracts NNUE-style HalfKP training features from JSONL chess samples.

    Parses a FEN board string into king-relative piece indices (from both
    the white and black king's perspective, each in [0, 640*10)) and
    converts an engine evaluation (centipawns or mate score) into a
    win-probability target via a logistic transform. Internal buffers are
    reused across calls for efficiency, so results are only valid until
    the next call.

    Attributes:
        white_indices: uint16[32] array of HalfKP feature indices from
            white king's perspective (INT16_MAX marks an unused slot).
        black_indices: uint16[32] array of HalfKP feature indices from
            black king's perspective (INT16_MAX marks an unused slot).
        pieces_char: List of piece characters scratch buffer, used
            internally during FEN parsing.
        pieces_sq: int8[32] array of piece square scratch buffer, used
            internally during FEN parsing.
        turn: uint8[1] array, 1 if white to move else 0.
        score: float32[1] array holding the win-probability target.
    """

    INT16_MAX = 65535
    PIECE_TO_TYPE = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "p": 5,
        "n": 6,
        "b": 7,
        "r": 8,
        "q": 9,
    }

    def __init__(self):
        self.white_indices = np.full(32, self.INT16_MAX, dtype=np.uint16)
        self.black_indices = np.full(32, self.INT16_MAX, dtype=np.uint16)
        self.pieces_char = [""] * 32
        self.pieces_sq = np.zeros(32, dtype=np.int32)
        self.turn = np.zeros(1, dtype=np.uint8)
        self.score = np.zeros(1, dtype=np.float32)

    def process_line(self, jsonl_bytes: bytes) -> tuple:
        """Parses one JSONL sample into model-ready features.

        Args:
            jsonl_bytes: A single JSON record (as bytes) with keys "fen"
                (FEN string) and "evals" (list of engine evaluations, each
                containing "pvs" with a "cp" or "mate" score).

        Returns:
            Tuple of (white_indices, black_indices, turn, score):

                - white_indices: uint16[32] HalfKP feature indices from
                  white king's perspective (INT16_MAX = unused slot).
                - black_indices: uint16[32] HalfKP feature indices from
                  black king's perspective (INT16_MAX = unused slot).
                - turn: uint8[1], 1 if white to move else 0.
                - score: float32[1], win probability in [0, 1] for the
                  side to move, derived from the evaluation.

        Note:
            Returned arrays are internal buffers reused on the next call;
            copy them if you need to retain values across calls.
        """

        raw_sample = orjson.loads(jsonl_bytes.decode("utf-8"))

        self.white_indices.fill(self.INT16_MAX)
        self.black_indices.fill(self.INT16_MAX)
        self.pieces_sq.fill(0)

        if not self._parse_fen(raw_sample["fen"]):
            return None

        self._parse_eval(raw_sample["evals"])

        return self.white_indices, self.black_indices, self.turn, self.score

    def _parse_fen(self, fen: str):
        fen_parts = fen.split(" ", 2)
        board = fen_parts[0]

        self.turn[0] = 1 if len(fen_parts) > 1 and fen_parts[1] == "w" else 0

        piece_count = 0
        wk_sq, bk_sq = -1, -1

        sq = 56
        for char in board:
            if char == "/":
                sq -= 16
            elif char.isdigit():
                sq += int(char)
            else:
                if char == "K":
                    wk_sq = sq
                elif char == "k":
                    bk_sq = sq
                else:
                    if piece_count >= 32:
                        return False

                    self.pieces_char[piece_count] = char
                    self.pieces_sq[piece_count] = sq
                    piece_count += 1
                sq += 1

        wk_offset = wk_sq * 640
        bk_offset = (bk_sq ^ 56) * 640

        for i in range(piece_count):
            char = self.pieces_char[i]
            sq = self.pieces_sq[i]
            p_type_w = self.PIECE_TO_TYPE[char]
            self.white_indices[i] = wk_offset + (p_type_w * 64) + sq

            p_type_b = p_type_w + 5 if p_type_w < 5 else p_type_w - 5
            self.black_indices[i] = bk_offset + (p_type_b * 64) + (sq ^ 56)

        return True

    def _parse_eval(self, evals: list):
        best_pv = evals[0]["pvs"][0]

        if "cp" in best_pv:
            score_cp = float(best_pv["cp"])
        elif "mate" in best_pv:
            mate_in = best_pv["mate"]

            if mate_in > 0:
                score_cp = 10000.0 - (mate_in * 10.0)
            else:
                score_cp = -10000.0 - (mate_in * 10.0)
        else:
            score_cp = 0.0

        if not self.turn[0]:
            score_cp = -score_cp

        self.score[0] = 1.0 / (1.0 + 10.0 ** (-score_cp / 400.0))
